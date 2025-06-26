from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
import torch
import numpy as np

# Load .npy arrays
# img1_path = '/scratch.global/wan01530/FMPlug/experiments/ae41v7ne/ground_truth_image.npy'
# img2_path = '/scratch.global/wan01530/FMPlug/experiments/ae41v7ne/reconstructed_image.npy'

img1_path = '/scratch.global/wan01530/FMPlug-SD3/workdir/flowdps/label/0000.npy'
img2_path = '/scratch.global/wan01530/FMPlug-SD3/workdir/flowdps/recon/0000.npy'

img1 = np.load(img1_path)
img2 = np.load(img2_path)

# Check size
if img1.shape != img2.shape:
    raise ValueError(f"Image dimensions do not match: {img1.shape} vs {img2.shape}")
print(img1.shape)
print(img2.shape)

# Ensure uint8 or float with correct range for PSNR and SSIM
if img1.dtype != np.uint8:
    img1_uint8 = (np.clip(img1, 0.0, 1.0) * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
    img2_uint8 = (np.clip(img2, 0.0, 1.0) * 255).astype(np.uint8) if img2.max() <= 1.0 else img2.astype(np.uint8)
else:
    img1_uint8 = img1
    img2_uint8 = img2

# Compute PSNR
psnr_value = compare_psnr(img1_uint8, img2_uint8, data_range=255)

# Optional: Compute SSIM
# ssim_value = compare_ssim(img1_uint8, img2_uint8, multichannel=True, data_range=255)

# Compute LPIPS (expects normalized float32 in [-1, 1])
# img1_lpips = torch.from_numpy(img1.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
# img2_lpips = torch.from_numpy(img2.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

img1_lpips = torch.from_numpy(img1.astype(np.float32))
img2_lpips = torch.from_numpy(img2.astype(np.float32))

if img1_lpips.max() > 1.0:
    img1_lpips = img1_lpips / 127.5 - 1.0
    img2_lpips = img2_lpips / 127.5 - 1.0
else:
    img1_lpips = img1_lpips * 2.0 - 1.0
    img2_lpips = img2_lpips * 2.0 - 1.0

loss_fn = lpips.LPIPS(net='vgg')
lpips_value = loss_fn(img1_lpips, img2_lpips).item()

# Report results
print(f"PSNR: {psnr_value:.2f} dB")
# print(f"SSIM: {ssim_value:.4f}")
print(f"LPIPS: {lpips_value:.4f}")
