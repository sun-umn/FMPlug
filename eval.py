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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clipiqa = CLIPIQA().to(device)
dists = DISTS().to(device)
musiq = pyiqa.create_metric('musiq', device=device)

root_dir = "/scratch.global/wan01530/FMPlug/experiment-DIP"
output_csv = "./performance_metrics-DIP.csv"
# target_size = (256, 256)  # Assuming target size for resizing
target_size = None  # Assuming target size for resizing

def load_image_tensor(filepath, target_size=None):
    """Loads a .npy image tensor and converts it to a PyTorch tensor."""
    img = np.load(filepath)
    # Assuming image is HWC or HW, convert to CHW and then to PyTorch tensor
    if img.ndim == 2:  # Grayscale
        img = np.expand_dims(img, axis=0) # Add channel dimension
    elif img.ndim == 3 and img.shape[2] in [1, 3]: # HWC
        img = np.transpose(img, (2, 0, 1)) # CHW
    elif  img.ndim == 4 and img.shape[1] in [1, 3]:
        img = np.squeeze(img)
        img = np.clip((img + 1) / 2, 0, 1)
    else:
        raise ValueError(f"Unsupported image dimensions: {img.shape}")
    
    # Normalize to [-1, 1] for LPIPS, assuming original data is [0, 255] or [0, 1]
    # If the data is already normalized to [0, 1], this will convert it to [-1, 1]
    # If the data is [0, 255], it will be normalized to [0, 1] first, then to [-1, 1]
    if img.max() > 1.0:
        img = img / 255.0
    
    img_tensor = torch.from_numpy(img).float()

    if target_size:
        # print(f"Original image tensor shape: {img_tensor.shape}")
        resize_transform = Resize(target_size)
        img_tensor = resize_transform(img_tensor)
        # print(f"Resized image tensor shape: {img_tensor.shape}")

    return img_tensor.unsqueeze(0) # Add batch dimension

def calculate_metrics(gt_path, recon_path, lpips_model, target_size=None): # Placeholder for H, W
    """Calculates LPIPS, PSNR, and SSIM between ground truth and reconstruction."""
    try:
        gt_img_tensor = load_image_tensor(gt_path, target_size=target_size).cuda()
        recon_img_tensor = load_image_tensor(recon_path, target_size=target_size).cuda()

        # The resizing is now handled in load_image_tensor, so this block is no longer needed
        # if gt_img_tensor.shape != recon_img_tensor.shape:
        #     from torchvision.transforms import Resize
        #     resize_transform = Resize(gt_img_tensor.shape[-2:])
        #     recon_img_tensor = resize_transform(recon_img_tensor)

        # LPIPS expects tensors in [-1, 1]
        lpips_val = lpips_model(gt_img_tensor * 2. - 1., recon_img_tensor * 2. - 1.).item()
        dists_val = dists(recon_img_tensor * 2. - 1, gt_img_tensor * 2. - 1).item()
        clipiqa_val = clipiqa(recon_img_tensor).item()
        musiq_val = musiq(recon_img_tensor).item()

        # PSNR and SSIM expect numpy arrays in [0, 1] or [0, 255]
        # Convert back to [0, 1] range for skimage metrics
        gt_np = (gt_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()).clip(0, 1)
        recon_np = (recon_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()).clip(0, 1)

        # Handle grayscale images for SSIM
        if gt_np.shape[2] == 1:
            gt_np = gt_np.squeeze(2)
            recon_np = recon_np.squeeze(2)
            multichannel = False
        else:
            multichannel = True

        psnr_val = peak_signal_noise_ratio(gt_np, recon_np, data_range=1.0)
        ssim_val = structural_similarity(gt_np, recon_np, data_range=1.0, multichannel=multichannel, channel_axis=-1 if multichannel else None)

        return lpips_val, clipiqa_val, dists_val, musiq_val, psnr_val, ssim_val
    except Exception as e:
        print(f"Error processing {gt_path} and {recon_path}: {e}")
        return None, None, None, None, None, None

def main():

    lpips_model = lpips.LPIPS(net='vgg').to(device)
    results = []

    # Iterate over time folders
    for time_name in os.listdir(root_dir):
        time_dir = os.path.join(root_dir, time_name)
        if not os.path.isdir(time_dir):
            continue

        for task_name in os.listdir(time_dir):
            task_dir = os.path.join(time_dir, task_name)
            if not os.path.isdir(task_dir):
                continue

            for dataset_name in os.listdir(task_dir):
                dataset_dir = os.path.join(task_dir, dataset_name)
                if not os.path.isdir(dataset_dir):
                    continue

                lpips_scores = []
                clipiqa_scores = []
                dists_scores = []
                musiq_scores = []
                psnr_scores = []
                ssim_scores = []

                image_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
                
                if not image_names:
                    print(f"No image folders found in {dataset_dir}. Skipping.")
                    continue

                for image_name in tqdm(image_names, desc=f"Processing {time_name}/{task_name}/{dataset_name}"):
                    image_dir = os.path.join(dataset_dir, image_name)
                    gt_path = os.path.join(image_dir, "gt.npy")
                    
                    reconstruction_files = [f for f in os.listdir(image_dir) if f.startswith("reconstruction_best_") and f.endswith(".npy")]
                    
                    if not os.path.exists(gt_path):
                        print(f"Warning: gt.npy not found in {image_dir}. Skipping image.")
                        continue
                    
                    if not reconstruction_files:
                        reconstruction_files = [f for f in os.listdir(image_dir) if f.startswith("reconstruction_last") and f.endswith(".npy")]
                    
                    if not reconstruction_files:
                        reconstruction_files = [f for f in os.listdir(image_dir) if f.startswith("reconstructed") and f.endswith(".npy")]
                    
                    if not reconstruction_files:
                        print(f"No reconstruction files found in {image_dir}. Skipping image.")
                        continue
                    
                    recon_path = os.path.join(image_dir, reconstruction_files[0])

                    lpips_val, clipiqa_val, dists_val, musiq_val, psnr_val, ssim_val = calculate_metrics(gt_path, recon_path, lpips_model, target_size)
                    
                    if lpips_val is not None:
                        lpips_scores.append(lpips_val)
                        clipiqa_scores.append(clipiqa_val)
                        dists_scores.append(dists_val)
                        musiq_scores.append(musiq_val)
                        psnr_scores.append(psnr_val)
                        ssim_scores.append(ssim_val)
                
                if lpips_scores:
                    avg_lpips = np.mean(lpips_scores)
                    avg_clipiqa = np.mean(clipiqa_scores)
                    avg_dists = np.mean(dists_scores)
                    avg_musiq = np.mean(musiq_scores)
                    avg_psnr = np.mean(psnr_scores)
                    avg_ssim = np.mean(ssim_scores)
                    results.append({
                        "time": time_name,
                        "task": task_name,
                        "dataset": dataset_name,
                        "psnr": avg_psnr,
                        "ssim": avg_ssim,
                        "lpips": avg_lpips,
                        "dists": avg_dists,
                        "clipiqa": avg_clipiqa,
                        "musiq": avg_musiq,
                    })
                else:
                    print(f"No valid scores for {time_name}/{task_name}/{dataset_name}. Skipping.")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Performance metrics saved to {output_csv}")
    else:
        print("No performance metrics to save.")


if __name__ == "__main__":
    main()