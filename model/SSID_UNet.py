from dataset.SIDD import SIDDSrgbTrainDataset, SIDDSrgbValidationDataset
from logger import Logger
from network.unet import UNet

import json
import numpy as np
import os
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio

def std(img, window_size=7):
    assert window_size % 2 == 1
    pad = window_size // 2

    # calculate std on the mean image of the color channels
    img = torch.mean(img, dim=1, keepdim=True)
    N, C, H, W = img.shape
    img = nn.functional.pad(img, [pad] * 4, mode='reflect')
    img = nn.functional.unfold(img, kernel_size=window_size)
    img = img.view(N, C, window_size * window_size, H, W)
    img = img - torch.mean(img, dim=2, keepdim=True)
    img = img * img
    img = torch.mean(img, dim=2, keepdim=True)
    img = torch.sqrt(img)
    img = img.squeeze(2)
    return img

def generate_alpha(input, lower=1, upper=5):
    N, C, H, W = input.shape
    ratio = input.new_ones((N, 1, H, W)) * 0.5
    input_std = std(input)
    ratio[input_std < lower] = torch.sigmoid((input_std - lower))[input_std < lower]
    ratio[input_std > upper] = torch.sigmoid((input_std - upper))[input_std > upper]
    ratio = ratio.detach()

    return ratio

class SSID_UNet():
    def __init__(self, cfg_path, BNN, LAN):
        cfg = {
            "dataroot": "../data",              # Path to data
            "logs_dir": "./logs/SSID_UNet",     # Path to logs
            "output_dir": "./output/SSID_UNet", # Path to ckpt outputs
            "load_from_ckpt": "",               # Path to ckpt to load from

            "patch_size": 256,                  # Image Crop Size

            "gpu": 0,            
            "batch_size": 16,
            "lr": 3e-4,
            "n_epochs": 1000,

            "print_every": 1,                   # Print loss to logs every ...
            "validate_every": 10,               # Validate model every ...
            "save_every": 50,                   # Save weights every ...

            "use_logs": True,
            "init_dataset": True                # Whether to initialize the datasets, set this to false if you only need inference
        }

        with open(cfg_path, "r") as f:
            user_cfg = json.load(f)

            for k, v in user_cfg.items():
                assert k in cfg, f"Unknown key {k} in config file"
                cfg[k] = v

        self.BNN = BNN
        self.LAN = LAN

        self.output_dir = cfg["output_dir"]

        self.logger = Logger(cfg["logs_dir"], disable=not cfg["use_logs"])
        self.logger.log("Initializing SSID UNet")
        self.logger.log("")
        self.logger.log("Arguments:")

        for k, v in cfg.items():
            self.logger.log("{0:<25}  {1}".format(k, v))
        self.logger.log("")

        self.batch_size = cfg["batch_size"]
        self.lr = cfg["lr"]
        self.start_epoch = 0
        self.n_epochs = cfg["n_epochs"]

        # To speed up set up for inference when datasets is not necessary
        if cfg["init_dataset"]:
            self.load_dataset(cfg["dataroot"], cfg["patch_size"])

        self.device = torch.device(f"cuda:{cfg['gpu']}" if cfg["gpu"] != -1 else "cpu")
        self.model = UNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_epochs)
        self.loss_fn = nn.L1Loss(reduction="mean")

        if cfg["load_from_ckpt"] != "":
            self.load_ckpt(cfg["load_from_ckpt"])

        model_summary = summary(self.model, (3, cfg["patch_size"], cfg["patch_size"]), verbose=0)
        self.logger.log(str(model_summary))
        self.logger.log("")

        self.print_every = cfg["print_every"]
        self.validate_every = cfg["validate_every"]
        self.save_every = cfg["save_every"]

    def load_dataset(self, dataroot, patch_size):
        self.logger.log("Loading Training Dataset...")
        self.logger.log("")
        self.train_dataset = SIDDSrgbTrainDataset(dataroot, patch_size=patch_size)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.logger.log("Loading Validation Dataset...")
        self.logger.log("")
        self.val_dataset = SIDDSrgbValidationDataset(dataroot)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=True)

    def load_ckpt(self, path_to_ckpt):
        """
        Assumes the checkpoints to follow two formats
        1. Pre-trained checkpoints given by the authors, consists only model state_dict.
        2. Checkpoints saved by us, consists of epoch, model, optimizer, and scheduler state_dict.
        """

        ckpt = torch.load(path_to_ckpt)

        if "model" not in ckpt:
            # Format 1
            self.model.load_state_dict(ckpt)
            self.logger.log("Loaded model from checkpoints")
            self.logger.log("")

        else:
            # Format 2
            self.start_epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])

            self.logger.log("Loaded model from checkpoints")
            self.logger.log("Loaded optimizer from checkpoints")
            self.logger.log("Loaded scheduler from checkpoints")
            self.logger.log(f"Starting from epoch {self.start_epoch}")
            self.logger.log("")

    def save_ckpt(self, path_to_ckpt, epoch=0):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }, path_to_ckpt)

        self.logger.log(f"Saved checkpoint to {path_to_ckpt}")
        self.logger.log("")

    def train(self):
        with tqdm(total=self.n_epochs * len(self.train_dataloader), desc="Training...") as p_bar:
            for epoch in range(self.start_epoch, self.n_epochs):
                self.model.train()

                for img_noisy, _ in self.train_dataloader:
                    img_noisy = img_noisy.to(self.device)
                    
                    img_bnn = self.BNN.inference(img_noisy, is_HWC=False, verbose=False)
                    img_bnn = torch.from_numpy(img_bnn).permute(0, 3, 1, 2).to(self.device)

                    img_lan = self.LAN.inference(img_noisy, is_HWC=False, verbose=False)
                    img_lan = torch.from_numpy(img_lan).permute(0, 3, 1, 2).to(self.device)

                    UNet = self.model(img_noisy)

                    alpha = generate_alpha(img_bnn)

                    loss = self.loss_fn(UNet * (1 - alpha), img_bnn * (1 - alpha)) + self.loss_fn(UNet * (1 - alpha), img_lan * (1 - alpha))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    p_bar.update(1)

                self.scheduler.step()

                if (epoch+1) % self.print_every == 0:
                    self.logger.log("Epoch {:04d}/{:04d}".format(epoch+1, self.n_epochs) + f"\tLoss {round(loss.item(), 6)}")

                if (epoch+1) % self.validate_every == 0:
                    _ = self.validate(log_psnr=True, verbose=False)

                if (epoch+1) % self.save_every == 0:
                    self.curr_epoch = epoch
                    self.save_ckpt(os.path.join(self.output_dir, f"epoch_{epoch+1}.pt"), epoch=epoch+1)

    def inference(self, imgs, is_HWC=True, verbose=True):
        """
        Returns the inference on a batch of numpy / tensor images in the shape of NHWC

        Parameters:
        --------------
        imgs: Input image batch, shape can be of HWC or NHWC
        """
        
        assert len(imgs.shape) == 3 or len(imgs.shape) == 4

        if len(imgs.shape) == 3:
            imgs = np.array([imgs])

        imgs_out = np.zeros(imgs.shape)
        if not is_HWC:
            imgs_out = np.transpose(imgs_out, (0, 2, 3, 1))

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(imgs.shape[0]), desc="Inferencing...", disable=not verbose):
                img_noisy = imgs[i, ...]

                # Clean Noisy Image
                if isinstance(img_noisy, np.ndarray):
                    # Numpy Array
                    if is_HWC:
                        img_noisy = np.transpose(img_noisy, (2, 0, 1))  # HWC to CHW

                    img_noisy = img_noisy[np.newaxis, ...]
                    img_noisy = torch.from_numpy(img_noisy)

                else:
                    # Tensor
                    if is_HWC:
                        img_noisy = img_noisy.permute(2, 0, 1)

                    img_noisy = img_noisy[None, ...]

                img_noisy = img_noisy.to(self.device)

                img_out = self.model(img_noisy)
                img_out = img_out.cpu().squeeze(0).permute(1, 2, 0).numpy()

                imgs_out[i, ...] = img_out

        return imgs_out

    def validate(self, log_psnr=True, verbose=True):
        psnrs, count = 0, 0

        self.model.eval()
        with torch.no_grad():
            for img_noisy, img_gt in tqdm(self.val_dataloader, desc="Validating...", disable=not verbose):
                img_noisy = img_noisy.to(self.device)

                img_out = self.model(img_noisy)
                img_out = img_out.cpu().squeeze(0).permute(1, 2, 0).numpy()
                img_out = np.uint8(img_out)

                img_gt = img_gt.squeeze(0).permute(1, 2, 0).numpy()

                psnr = peak_signal_noise_ratio(img_out, img_gt, data_range=255)
                psnrs += psnr
                count += 1

        if log_psnr:
            self.logger.log("")
            self.logger.log(f"PSNR {round(psnrs / count, 6)}")
            self.logger.log("")

        return psnrs / count
    
    def validate_custom(self, imgs, imgs_gt, is_HWC=True):
        assert len(imgs.shape) == 3 or len(imgs.shape == 4), f"Invalid Noisy Image Shape {imgs.shape}"
        assert len(imgs_gt.shape) == 3 or len(imgs_gt.shape == 4), f"Invalid Ground Truth Image Shape {imgs_gt.shape}"

        assert type(imgs) == type(imgs_gt), \
            f"Noisy and Ground Truth Images should be both the same, found {type(imgs)} and {type(imgs_gt)}"

        assert isinstance(imgs, np.ndarray) or torch.is_tensor(imgs), \
            f"Noisy and Ground Truth Images should be either both numpy arrays or tensors, found {type(imgs)} and {type(imgs_gt)}"

        if len(imgs.shape) == 3:
            imgs = imgs[None, ...]

        if len(imgs_gt.shape) == 3:
            imgs_gt = imgs_gt[None, ...]

        psnrs, count = 0, imgs.shape[0]

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(imgs.shape[0]), desc="Validating..."):
                img_noisy = imgs[i, ...]
                img_gt = imgs_gt[i, ...]

                # Clean Noisy Image
                if isinstance(img_noisy, np.ndarray):
                    # Numpy Array
                    if is_HWC:
                        img_noisy = np.transpose(img_noisy, (2, 0, 1))

                    img_noisy = img_noisy[np.newaxis, ...]
                    img_noisy = torch.from_numpy(img_noisy)

                else:
                    # Tensor
                    if is_HWC:
                        img_noisy = img_noisy.permute(2, 0, 1)

                    img_noisy = img_noisy[None, ...]

                img_noisy = img_noisy.to(self.device)
            
                # Inference
                img_out = self.model(img_noisy)
                img_out = img_out.cpu().squeeze(0).permute(1, 2, 0).numpy()
                img_out = np.uint8(img_out)

                # Clean Ground Truth Image
                if isinstance(img_gt, np.ndarray):
                    # Numpy Array
                    if not is_HWC:
                        img_gt = np.transpose(img_gt, (1, 2, 0))

                else:
                    # Tensor
                    if not is_HWC:
                        img_gt = img_gt.permute(1, 2, 0)

                # Compute PSNR
                psnr = peak_signal_noise_ratio(img_out, img_gt, data_range=255)
                psnrs += psnr

        return psnrs / count