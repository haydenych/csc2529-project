from dataset.SIDD import SIDDSrgbTrainDataset, SIDDSrgbValidationDataset
from logger import Logger
from network.ssid_bnn import SSID_BNN

import argparse
import os
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio


def train(args, model, device, logger):
    dataset = SIDDSrgbTrainDataset(args.dataroot, args.patch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    loss_fn = nn.L1Loss(reduction="mean")

    with tqdm(total=args.n_epochs * len(dataloader), position=0, leave=True) as p_bar:
        for epoch in range(args.n_epochs):
            model.train()

            for img_noisy, _ in dataloader:
                img_noisy = img_noisy.to(device)
                BNN = model(img_noisy)
                loss = loss_fn(BNN, img_noisy)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                p_bar.update(1)

            scheduler.step()

            if (epoch+1) % args.print_every == 0:
                logger.log("Epoch {:04d}/{:04d}".format(epoch+1, args.n_epochs) + f"\tLoss {round(loss.item(), 6)}")

            if (epoch+1) % args.validate_every == 0:
                psnr = test(args, model, device)
                logger.log("")
                logger.log("Epoch {:04d}/{:04d}".format(epoch+1, args.n_epochs) + f"\tPSNR {round(psnr, 6)}")
                logger.log("")

            if (epoch+1) % args.save_every == 0:
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)

                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }, os.path.join(args.output_dir, f"epoch_{epoch+1}.pt"))


def test(args, model, device):
    dataset = SIDDSrgbValidationDataset(args.dataroot)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    psnrs, count = 0, 0

    model.eval()
    with torch.no_grad():
        for img_noisy, img_gt in dataloader:
            img_noisy = img_noisy.to(device)
            BNN = model(img_noisy)

            BNN = BNN.cpu().squeeze(0).permute(1, 2, 0).numpy()
            img_gt = img_gt.squeeze(0).permute(1, 2, 0).numpy()
            psnr = peak_signal_noise_ratio(BNN, img_gt, data_range=255)
            psnrs += psnr
            count += 1

    return psnrs / count


def main(args):
    logger = Logger(os.path.join(args.path_to_logs, "ssid_bnn"))
    logger.log("python3 finetune.py")
    logger.log("\nArguments:")

    for k, v in vars(args).items():
        logger.log("{0:<25}  {1}".format(k, v))
    logger.log("")

    device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")
    model = SSID_BNN(blindspot=9).to(device)

    model_summary = summary(model, (3, args.patch_size, args.patch_size), verbose=0)
    logger.log(str(model_summary))
    logger.log("")

    train(args, model, device, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0, help="GPU Device ID, -1 for CPU")

    parser.add_argument('--dataroot', type=str, default='../data', help="Path to datasets")
    parser.add_argument('--output-dir', type=str, default='./output', help="Path to outputs")
    parser.add_argument('--path-to-logs', type=str, default='./output/logs', help="Path to logs")

    parser.add_argument('--batch-size', type=int, default=4, help="Batch size")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--n-epochs', type=int, default=1000, help="Number of epochs")

    parser.add_argument('--patch-size', type=int, default=256, help="Patch size")

    parser.add_argument('--validate-every', type=int, default=10)
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--save-every', type=int, default=50)

    args = parser.parse_args()

    main(args)