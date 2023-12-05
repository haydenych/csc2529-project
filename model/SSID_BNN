from dataset.SIDD import SIDDSrgbTrainDataset, SIDDSrgbValidationDataset
from logger import Logger
from network.ssid_bnn import SSID_BNN

import json
import os
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio

class SSID_BNN():
    def __init__(self, cfg_path):
        cfg = {
            "dataroot": "../data",              # Path to data
            "logs_dir": "./logs/SSID_BNN",      # Path to logs
            "output_dir": "./output",           # Path to ckpt outputs

            "patch_size": 256,                  # Image Crop Size

            "gpu": 0,            
            "batch_size": 4,
            "lr": 3e-4,
            "n_epochs": 1000,

            "print_every": 1,                   # Print loss to logs every ...
            "validate_every": 10,               # Validate model every ...
            "save_every": 50                    # Save weights every ...
        }

        # TODO: Implement load from ckpt

        cfg_ = json.loads(cfg_path)
        for k, v in cfg_.items():
            assert(k in cfg, f"Unknown key {k} in config file")
            cfg[k] = v

        self.batch_size = cfg["batch_size"]
        self.lr = cfg["lr"]
        self.start_epoch = 0
        self.n_epochs = cfg["n_epochs"]

        self.train_dataset = SIDDSrgbTrainDataset(cfg["dataroot"], patch_size=cfg["patch_size"])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        self.val_dataset = SIDDSrgbValidationDataset(cfg["dataroot"])
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=True)

        self.device = torch.device(f"cuda:{cfg['gpu']}" if cfg["gpu"] != -1 else "cpu")
        self.model = SSID_BNN(blindspot=9).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_epochs)
        self.loss_fn = nn.L1Loss(reduction="mean")

        self.logger = Logger(cfg["logs_dir"])

        # Add logs for cfg and model summary

    def train(self):
        with tqdm(total=self.n_epochs * len(self.train_dataloader)) as p_bar:
            for epoch in range(self.start_epoch, self.n_epochs):
                self.model.train()

                for img_noisy, _ in self.dataloader:
                    img_noisy = img_noisy.to(self.device)
                    BNN = self.model(img_noisy)
                    loss = self.loss_fn(BNN, img_noisy)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    p_bar.update(1)

                self.scheduler.step()

                if (epoch+1) % self.print_every == 0:
                    self.logger.log("Epoch {:04d}/{:04d}".format(epoch+1, self.n_epochs) + f"\tLoss {round(loss.item(), 6)}")

                if (epoch+1) % self.validate_every == 0:
                    psnr = self.validate()
                    self.logger.log("")
                    self.logger.log("Epoch {:04d}/{:04d}".format(epoch+1, self.n_epochs) + f"\tPSNR {round(psnr, 6)}")
                    self.logger.log("")

                if (epoch+1) % self.save_every == 0:
                    if not os.path.exists(self.output_dir):
                        os.makedirs(self.output_dir)

                    torch.save({
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict()
                    }, os.path.join(self.output_dir, f"epoch_{epoch+1}.pt"))

    def validate(self):
        psnrs, count = 0, 0

        self.model.eval()
        with torch.no_grad():
            for img_noisy, img_gt in self.val_dataloader:
                img_noisy = img_noisy.to(self.device)
                img_out = self.model(img_noisy)

                img_out = img_out.cpu().squeeze(0).permute(1, 2, 0).numpy()
                img_gt = img_gt.squeeze(0).permute(1, 2, 0).numpy()

                psnr = peak_signal_noise_ratio(img_out, img_gt, data_range=255)
                psnrs += psnr
                count += 1

        return psnrs / count