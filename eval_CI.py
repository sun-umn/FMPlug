import argparse
import os
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from piq import CLIPIQA, DISTS
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import Resize
import pyiqa
from scipy import stats
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clipiqa = CLIPIQA().to(device)
dists = DISTS().to(device)
musiq = pyiqa.create_metric('musiq', device=device)

root_dir = "/scratch.global/wan01530/FMPlug/experiment-DFlow-noiseless-3/2025-09-16 00:50:20"
output_csv = "./performance_metrics_DFlow-noiseless-3-CI.csv"
target_size = None  # Optional resizing, e.g. (256, 256)


def load_image_tensor(filepath, target_size=None):
    """Loads a .npy image tensor and converts it to a PyTorch tensor."""
    img = np.load(filepath)
    if img.ndim == 2:  # Grayscale
        img = np.expand_dims(img, axis=0)  # C, H, W
    elif img.ndim == 3 and img.shape[2] in [1, 3]:  # HWC
        img = np.transpose(img, (2, 0, 1))  # CHW
    elif img.ndim == 4 and img.shape[1] in [1, 3]:
        img = np.squeeze(img)
        img = np.clip((img + 1) / 2, 0, 1)
    else:
        raise ValueError(f"Unsupported image dimensions: {img.shape}")

    if img.max() > 1.0:
        img = img / 255.0

    img_tensor = torch.from_numpy(img).float()

    if target_size:
        resize_transform = Resize(target_size)
        img_tensor = resize_transform(img_tensor)

    return img_tensor.unsqueeze(0)  # Add batch dimension


def calculate_metrics(gt_path, recon_path, lpips_model, target_size=None):
    """Calculates LPIPS, PSNR, SSIM, CLIPIQA, DISTS, MUSIQ between GT and reconstruction."""
    try:
        gt_img_tensor = load_image_tensor(gt_path, target_size=target_size).to(device)
        recon_img_tensor = load_image_tensor(recon_path, target_size=target_size).to(device)

        lpips_val = lpips_model(gt_img_tensor * 2. - 1., recon_img_tensor * 2. - 1.).item()
        dists_val = dists(recon_img_tensor * 2. - 1, gt_img_tensor * 2. - 1).item()
        clipiqa_val = clipiqa(recon_img_tensor).item()
        musiq_val = musiq(recon_img_tensor).item()

        # Convert to numpy [0, 1]
        gt_np = (gt_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()).clip(0, 1)
        recon_np = (recon_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()).clip(0, 1)

        if gt_np.shape[2] == 1:
            gt_np = gt_np.squeeze(2)
            recon_np = recon_np.squeeze(2)
            multichannel = False
        else:
            multichannel = True

        psnr_val = peak_signal_noise_ratio(gt_np, recon_np, data_range=1.0)
        ssim_val = structural_similarity(
            gt_np, recon_np, data_range=1.0,
            channel_axis=-1 if multichannel else None
        )

        return lpips_val, clipiqa_val, dists_val, musiq_val, psnr_val, ssim_val
    except Exception as e:
        print(f"Error processing {gt_path} and {recon_path}: {e}")
        return None, None, None, None, None, None


def compute_mean_ci(values, confidence=0.95):
    """Compute mean and half-width of 95% confidence interval for a list of values."""
    arr = np.array(values)
    n = len(arr)
    if n < 2:
        return np.mean(arr), 0.0  # No CI if only one sample
    mean = np.mean(arr)
    sem = stats.sem(arr)
    h = stats.t.ppf((1 + confidence) / 2., n - 1) * sem  # half-width
    return mean, h


def main():
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    # Collect metrics across trials
    metrics_dict = defaultdict(lambda: defaultdict(list))
    # key = (task, dataset, image_id)

    for trial in os.listdir(root_dir):
        trial_dir = os.path.join(root_dir, trial)
        if not os.path.isdir(trial_dir):
            continue

        for task_name in os.listdir(trial_dir):
            task_dir = os.path.join(trial_dir, task_name)
            if not os.path.isdir(task_dir):
                continue

            for dataset_name in os.listdir(task_dir):
                dataset_dir = os.path.join(task_dir, dataset_name)
                if not os.path.isdir(dataset_dir):
                    continue

                image_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
                if not image_names:
                    continue

                for image_name in image_names:
                    image_dir = os.path.join(dataset_dir, image_name)
                    gt_path = os.path.join(image_dir, "gt.npy")

                    reconstruction_files = [f for f in os.listdir(image_dir) if f.startswith("reconstruction_best_") and f.endswith(".npy")]
                    if not os.path.exists(gt_path):
                        continue
                    if not reconstruction_files:
                        reconstruction_files = [f for f in os.listdir(image_dir) if f.startswith("reconstruction_last") and f.endswith(".npy")]
                    if not reconstruction_files:
                        reconstruction_files = [f for f in os.listdir(image_dir) if f.startswith("reconstructed") and f.endswith(".npy")]
                    if not reconstruction_files:
                        continue

                    recon_path = os.path.join(image_dir, reconstruction_files[0])

                    lpips_val, clipiqa_val, dists_val, musiq_val, psnr_val, ssim_val = calculate_metrics(
                        gt_path, recon_path, lpips_model, target_size
                    )

                    if lpips_val is not None:
                        key = (task_name, dataset_name, image_name)
                        metrics_dict[key]["lpips"].append(lpips_val)
                        metrics_dict[key]["clipiqa"].append(clipiqa_val)
                        metrics_dict[key]["dists"].append(dists_val)
                        metrics_dict[key]["musiq"].append(musiq_val)
                        metrics_dict[key]["psnr"].append(psnr_val)
                        metrics_dict[key]["ssim"].append(ssim_val)

    # Aggregate across trials
    results = []
    for (task_name, dataset_name, image_name), metrics in metrics_dict.items():
        avg_lpips, lpips_ci = compute_mean_ci(metrics["lpips"])
        avg_clipiqa, clipiqa_ci = compute_mean_ci(metrics["clipiqa"])
        avg_dists, dists_ci = compute_mean_ci(metrics["dists"])
        avg_musiq, musiq_ci = compute_mean_ci(metrics["musiq"])
        avg_psnr, psnr_ci = compute_mean_ci(metrics["psnr"])
        avg_ssim, ssim_ci = compute_mean_ci(metrics["ssim"])

        results.append({
            "task": task_name,
            "dataset": dataset_name,
            "image_id": image_name,
            "psnr_mean": avg_psnr,
            "psnr_ci95": psnr_ci,
            "ssim_mean": avg_ssim,
            "ssim_ci95": ssim_ci,
            "lpips_mean": avg_lpips,
            "lpips_ci95": lpips_ci,
            "dists_mean": avg_dists,
            "dists_ci95": dists_ci,
            "clipiqa_mean": avg_clipiqa,
            "clipiqa_ci95": clipiqa_ci,
            "musiq_mean": avg_musiq,
            "musiq_ci95": musiq_ci,
        })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Performance metrics with 95% CI (mean ± ci95) across trials saved to {output_csv}")
    else:
        print("No performance metrics to save.")


if __name__ == "__main__":
    main()
