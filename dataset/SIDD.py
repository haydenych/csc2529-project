"""
This code is modified from Spatially Adaptive SSID.

We observe slightly better performance with training inputs in [0, 255] range than that in [0, 1],
so we follow AP-BSN that do not normalize the input image from [0, 255] to [0, 1].
"""

import glob
import numpy as np
import os
import random
import scipy

from PIL import Image
from torch.utils.data import Dataset

def aug_np3(img, flip_h, flip_w, transpose):
    if flip_h:
        img = img[:, ::-1, :]
    if flip_w:
        img = img[:, :, ::-1]
    if transpose:
        img = np.transpose(img, (0, 2, 1))

    return img


def crop_np3(img, patch_size, position_h, position_w):
    return img[:, position_h:position_h+patch_size, position_w:position_w+patch_size]


class SIDDSrgbTrainDataset(Dataset):
    def __init__(self, dataroot, patch_size):
        self.imgs = []

        noisy_img_paths = sorted(glob.glob(os.path.join(dataroot, "SIDD/SIDD_Medium_Srgb/Data/*/*_NOISY_SRGB_*.PNG")))
        for noisy_img_path in noisy_img_paths:
            img_noisy = Image.open(noisy_img_path)
            img_gt = Image.open(noisy_img_path.replace("NOISY", "GT"))

            self.imgs.append(
                {
                    "NOISY": np.array(img_noisy),
                    "GT": np.array(img_gt)
                }
            )

            img_noisy.close()
            img_gt.close()

        self.patch_size = patch_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_noisy = self.imgs[index]["NOISY"]
        img_noisy = np.array(img_noisy, dtype=np.float32)
        img_noisy = np.transpose(img_noisy, (2, 0, 1))  # HWC to CHW

        img_gt = self.imgs[index]["GT"]
        img_gt = np.array(img_gt, dtype=np.float32)
        img_gt = np.transpose(img_gt, (2, 0, 1))  # HWC to CHW

        img_noisy, img_gt = self._crop(img_noisy, img_gt)
        img_noisy, img_gt = self._augment(img_noisy, img_gt)

        # Fix problem of not supporting negative strides
        # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
        img_noisy = img_noisy.copy()
        img_gt = img_gt.copy()

        return img_noisy, img_gt

    def _crop(self, img_noisy, img_gt):
        C, H, W = img_noisy.shape
        position_H = random.randint(0, H - self.patch_size)
        position_W = random.randint(0, W - self.patch_size)

        patch_noisy = crop_np3(img_noisy, self.patch_size, position_H, position_W)
        patch_gt = crop_np3(img_gt, self.patch_size, position_H, position_W)

        return patch_noisy, patch_gt

    def _augment(self, img_noisy, img_gt):
        flip_h = random.random() > 0.5
        flip_w = random.random() > 0.5
        transpose = random.random() > 0.5

        img_noisy = aug_np3(img_noisy, flip_h, flip_w, transpose)
        img_gt = aug_np3(img_gt, flip_h, flip_w, transpose)

        return img_noisy, img_gt


class SIDDSrgbValidationDataset(Dataset):
    def __init__(self, dataroot):
        mat = scipy.io.loadmat(os.path.join(dataroot, "SIDD/SIDD_Validation/ValidationNoisyBlocksSrgb.mat"))
        self.noisy_block = mat["ValidationNoisyBlocksSrgb"]

        mat = scipy.io.loadmat(os.path.join(dataroot, "SIDD/SIDD_Validation/ValidationGtBlocksSrgb.mat"))
        self.gt_block = mat["ValidationGtBlocksSrgb"]

        self.n = self.noisy_block.shape[0]
        self.k = self.noisy_block.shape[1]

    def __getitem__(self, index):
        index_n = index // self.k
        index_k = index % self.k

        img_noisy = self.noisy_block[index_n, index_k]
        img_noisy = np.float32(img_noisy)
        img_noisy = np.transpose(img_noisy, (2, 0, 1))  # HWC to CHW

        img_gt = self.gt_block[index_n, index_k]
        img_gt = np.float32(img_gt)
        img_gt = np.transpose(img_gt, (2, 0, 1))  # HWC to CHW

        return img_noisy, img_gt

    def __len__(self):
        return self.n * self.k


class SIDDSrgbBenchmarkDataset(Dataset):
    def __init__(self, dataroot):
        mat = scipy.io.loadmat(os.path.join(dataroot, "SIDD/SIDD_Benchmark/BenchmarkNoisyBlocksSrgb.mat"))
        self.noisy_block = mat["BenchmarkNoisyBlocksSrgb"]

        self.n = self.noisy_block.shape[0]
        self.k = self.noisy_block.shape[1]

    def __getitem__(self, index):
        index_n = index // self.k
        index_k = index % self.k

        img_noisy = self.noisy_block[index_n, index_k]
        img_noisy = np.float32(img_noisy)
        img_noisy = np.transpose(img_noisy, (2, 0, 1))  # HWC to CHW


        return img_noisy

    def __len__(self):
        return self.n * self.k
