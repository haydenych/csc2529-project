import argparse
import numpy as np
import os

from glob import glob
from multiprocessing import Pool, current_process
from PIL import Image
from tqdm import tqdm


def correlation(patch):
    """
    Returns the spatial correlation of an image patch.
    Computed using the pearson correlation coefficient between the center pixel and the neighbor pixel.    
    """

    d = patch.shape[0] // 2
    mid = patch[d, d, :]

    patch_shifted = patch - np.repeat(patch.mean(axis=2)[..., np.newaxis], 3, axis=2)
    mid_shifted = mid - np.mean(mid)

    patch_numer = patch_shifted
    mid_numer = np.ones_like(patch) * mid_shifted

    numer = np.sum(patch_numer * mid_numer, axis=2)

    patch_norm = np.linalg.norm(patch_shifted, axis=2)
    mid_norm = np.ones((2*d+1, 2*d+1)) * np.linalg.norm(mid_shifted)

    denom = patch_norm * mid_norm

    return numer / denom


def spatial_corr(x, d, p_bar_index):
    """
    Returns the spatial correlation with relative distance of d for a single image.
    """

    sum = np.zeros((2*d+1, 2*d+1))
    cnt = np.zeros((2*d+1, 2*d+1))

    pad = np.zeros((x.shape[0] + 2*d, x.shape[1] + 2*d, x.shape[2]))
    for c in range(x.shape[2]):
        pad[:, :, c] = np.pad(x[:, :, c], d)

    with tqdm(total=x.shape[0]*x.shape[1], position=p_bar_index, leave=False) as p_bar:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                patch = pad[i:i+2*d+1, j:j+2*d+1, :]

                corr = correlation(patch)

                sum_ = np.nan_to_num(corr, copy=True)
                cnt_ = np.where(np.isnan(corr), 0, 1)

                sum += sum_
                cnt += cnt_

                p_bar.update(1)

    return sum, cnt


def mp_func(args):
    np.seterr(divide="ignore", invalid="ignore")

    img_noise_path = args[0]
    img_clean_path = args[1]
    d = args[2]

    x = np.array(Image.open(img_noise_path), dtype=np.int16) - np.array(Image.open(img_clean_path), dtype=np.int16)
    return spatial_corr(x, d, current_process()._identity[0])


def main(args):
    img_all = glob(os.path.join(args.dataroot, "*/*.PNG"), recursive=True)
    img_noise_path_list = sorted([x for x in img_all if "NOISY" in x])
    img_clean_path_list = sorted([x for x in img_all if "GT" in x])

    assert(len(img_noise_path_list) == len(img_clean_path_list))
    assert(len(img_noise_path_list) > 0)

    if args.num_samples == -1 or args.num_samples > len(img_noise_path_list):
        args.num_samples = len(img_noise_path_list)

    mp_args = zip(img_noise_path_list[:args.num_samples], img_clean_path_list[:args.num_samples], [args.d] * args.num_samples)

    with Pool(processes=args.num_procs, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        res = list(tqdm(p.imap(mp_func, mp_args), total=args.num_samples))
    res = np.array(res)

    # Noise Correlation Map from all samples
    noise_map_total_ = res.sum(axis=0)
    noise_map_total = noise_map_total_[0,:,:] / noise_map_total_[1,:,:]

    # Noise Correlation Map for each individual sample
    noise_map_individual = res[:,0,:,:] / res[:,1,:,:]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, f"noise_total_{args.d}_{args.num_samples}.npy"), "wb") as f:
        np.save(f, noise_map_total)

    with open(os.path.join(args.output_dir, f"noise_individual_{args.d}_{args.num_samples}.npy"), "wb") as f:
        np.save(f, noise_map_individual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--dataroot", type=str, default="../../data/SIDD/SIDD_Medium_Srgb/Data")
    parser.add_argument("-o", "--output-dir", type=str, default="./output")
    parser.add_argument("-n", "--num-samples", type=int, default=-1, help="Number of samples to generate the noise map, use -1 for all")
    parser.add_argument("-d", "--d", type=int, default=10)
    parser.add_argument("-p", "--num-procs", type=int, default=24)

    args = parser.parse_args()
    main(args)



