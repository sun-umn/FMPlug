from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
import torch
import numpy as np
import os
from DISTS_pytorch import DISTS
from pytorch_fid.fid_score import calculate_fid_given_paths
from PIL import Image
from torchmetrics.image.quality_assessment import NIQE # For NIQE

# --- Configuration ---
# Manually set the base experiment directory here.
# Example: base_experiment_dir = '/scratch.global/wan01530/FMPlug/experiments/my_experiment_run/'
base_experiment_dir = '/scratch.global/wan01530/FMPlug/experiments/xxxx/' # REPLACE 'xxxx' with your actual experiment ID

# --- Initialize Metric Accumulators ---
psnr_scores = []
ssim_scores = []
lpips_scores = []
dists_scores = []

# Lists to store images for FID calculation
gt_images_for_fid = []
recon_images_for_fid = []

# Lists to store reconstructed images for no-reference metrics
recon_images_for_no_ref_metrics = []
niqe_scores = [] # For NIQE scores

# Initialize LPIPS, DISTS, and NIQE models
lpips_loss_fn = lpips.LPIPS(net='vgg').cuda() # Use .cuda() if GPU is available
dists_loss_fn = DISTS().cuda() # Use .cuda() if GPU is available
niqe_metric = NIQE(data_range=255.0).cuda() # Initialize NIQE metric, assuming data_range 0-255

# --- Helper Functions ---
def preprocess_image_for_lpips_dists(img_np):
    """
    Preprocesses a NumPy image array for LPIPS and DISTS calculation.
    Converts to torch.Tensor, permutes to (N, C, H, W), and normalizes to [-1, 1].
    Assumes input is (H, W, C) or (H, W) and converts to (C, H, W) if needed.
    """
    if img_np.ndim == 2: # Grayscale image
        img_np = np.expand_dims(img_np, axis=-1) # Add channel dimension
    if img_np.shape[-1] == 1: # Grayscale, convert to 3 channels for LPIPS/DISTS
        img_np = np.repeat(img_np, 3, axis=-1)

    img_tensor = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

    # Normalize to [-1, 1]
    if img_tensor.max() > 1.0: # Assuming range [0, 255]
        img_tensor = img_tensor / 127.5 - 1.0
    else: # Assuming range [0, 1]
        img_tensor = img_tensor * 2.0 - 1.0
    return img_tensor.cuda() # Use .cuda() if GPU is available

def preprocess_image_for_fid(img_np):
    """
    Preprocesses a NumPy image array for FID calculation.
    Converts to uint8 and ensures 3 channels.
    """
    if img_np.dtype != np.uint8:
        # Clip to [0, 1] if float, then scale to [0, 255] and convert to uint8
        if img_np.max() <= 1.0:
            img_np = (np.clip(img_np, 0.0, 1.0) * 255).astype(np.uint8)
        else: # Assuming range [0, 255] or similar, just convert
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    if img_np.ndim == 2: # Grayscale image
        img_np = np.expand_dims(img_np, axis=-1) # Add channel dimension
    if img_np.shape[-1] == 1: # Grayscale, convert to 3 channels
        img_np = np.repeat(img_np, 3, axis=-1)
    
    return img_np

# --- Main Processing Loop ---
for subdir in os.listdir(base_experiment_dir):
    subdir_path = os.path.join(base_experiment_dir, subdir)
    if os.path.isdir(subdir_path):
        gt_path = os.path.join(subdir_path, 'ground_truth_image.npy')
        recon_path = os.path.join(subdir_path, 'reconstructed_image.npy')

        if os.path.exists(gt_path) and os.path.exists(recon_path):
            print(f"Processing: {subdir}")
            img_gt = np.load(gt_path)
            img_recon = np.load(recon_path)

            if img_gt.shape != img_recon.shape:
                print(f"Warning: Image dimensions do not match for {subdir}. Skipping. {img_gt.shape} vs {img_recon.shape}")
                continue

            # Ensure uint8 for PSNR and SSIM
            img_gt_uint8 = (np.clip(img_gt, 0.0, 1.0) * 255).astype(np.uint8) if img_gt.max() <= 1.0 else img_gt.astype(np.uint8)
            img_recon_uint8 = (np.clip(img_recon, 0.0, 1.0) * 255).astype(np.uint8) if img_recon.max() <= 1.0 else img_recon.astype(np.uint8)

            # PSNR
            psnr_value = compare_psnr(img_gt_uint8, img_recon_uint8, data_range=255)
            psnr_scores.append(psnr_value)

            # SSIM
            # SSIM expects (H, W) or (H, W, C). If C is 1, multichannel=False. If C > 1, multichannel=True.
            # Ensure images are 3D (H, W, C) for multichannel=True, or 2D (H, W) for grayscale.
            if img_gt_uint8.ndim == 2: # Grayscale
                ssim_value = compare_ssim(img_gt_uint8, img_recon_uint8, data_range=255)
            else: # Color
                ssim_value = compare_ssim(img_gt_uint8, img_recon_uint8, multichannel=True, data_range=255)
            ssim_scores.append(ssim_value)

            # LPIPS and DISTS
            img_gt_lpips_dists = preprocess_image_for_lpips_dists(img_gt)
            img_recon_lpips_dists = preprocess_image_for_lpips_dists(img_recon)
            
            lpips_value = lpips_loss_fn(img_gt_lpips_dists, img_recon_lpips_dists).item()
            lpips_scores.append(lpips_value)

            dists_value = dists_loss_fn(img_gt_lpips_dists, img_recon_lpips_dists).item()
            dists_scores.append(dists_value)

            # Prepare images for FID (need to save to temporary directory for pytorch-fid)
            # For simplicity, we'll collect numpy arrays and convert to PIL images later if needed for a temp dir.
            gt_images_for_fid.append(preprocess_image_for_fid(img_gt))
            recon_images_for_fid.append(preprocess_image_for_fid(img_recon))

            # Prepare images for no-reference metrics (NIQE, MUSIQ, MANIQA, CLIPIQA)
            # For NIQE, we need a torch tensor in (N, C, H, W) format, normalized to [0, 255]
            # The preprocess_image_for_fid already returns uint8 (0-255) and 3 channels.
            # We just need to convert to tensor and permute.
            img_recon_tensor_niqe = torch.from_numpy(preprocess_image_for_fid(img_recon)).permute(2, 0, 1).unsqueeze(0).cuda()
            niqe_scores.append(niqe_metric(img_recon_tensor_niqe).item())

            # For MUSIQ, MANIQA, CLIPIQA, we collect the raw numpy arrays
            recon_images_for_no_ref_metrics.append(preprocess_image_for_fid(img_recon))

# --- Calculate FID ---
# pytorch-fid requires paths to directories of images.
# We need to create temporary directories and save the images there.
import tempfile
import shutil

fid_value = float('nan') # Initialize FID to NaN in case of error or no images

if gt_images_for_fid and recon_images_for_fid:
    with tempfile.TemporaryDirectory() as gt_temp_dir, \
         tempfile.TemporaryDirectory() as recon_temp_dir:
        
        for i, img_np in enumerate(gt_images_for_fid):
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(gt_temp_dir, f'gt_{i:04d}.png'))
        
        for i, img_np in enumerate(recon_images_for_fid):
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(recon_temp_dir, f'recon_{i:04d}.png'))

        paths = [gt_temp_dir, recon_temp_dir]
        # Assuming default device (cuda if available, else cpu) and batch_size
        fid_value = calculate_fid_given_paths(paths, batch_size=50, device='cuda' if torch.cuda.is_available() else 'cpu', dims=2048)
else:
    print("Not enough images to calculate FID.")

# --- Calculate No-Reference Metrics ---
# NIQE is calculated per image and accumulated in niqe_scores.
# MUSIQ, MANIQA, CLIPIQA require specific pre-trained models and are not directly available
# in standard libraries without additional setup. They are kept as placeholders.
musiq_value = float('nan')
maniqa_value = float('nan')
clipiqa_value = float('nan')

if not niqe_scores:
    print("\nNot enough reconstructed images to calculate NIQE.")
    niqe_value = float('nan')
else:
    niqe_value = np.mean(niqe_scores)

print("\n--- No-Reference Metrics (Additional Implementations Needed) ---")
print("MUSIQ, MANIQA, CLIPIQA: These metrics typically require specific pre-trained deep learning models and their associated libraries, which are not readily available for direct integration without further setup or installation of specialized packages. They are currently placeholders.")


# --- Report Results ---
print("\n--- Average Metrics ---")
if psnr_scores:
    print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
else:
    print("PSNR: N/A (no images processed)")

if ssim_scores:
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
else:
    print("SSIM: N/A (no images processed)")

if lpips_scores:
    print(f"Average LPIPS: {np.mean(lpips_scores):.4f}")
else:
    print("LPIPS: N/A (no images processed)")

if dists_scores:
    print(f"Average DISTS: {np.mean(dists_scores):.4f}")
else:
    print("DISTS: N/A (no images processed)")

print(f"FID: {fid_value:.4f}")

print(f"Average NIQE: {niqe_value:.4f}")
print(f"MUSIQ: {musiq_value:.4f} (Placeholder - requires model)")
print(f"MANIQA: {maniqa_value:.4f} (Placeholder - requires model)")
print(f"CLIPIQA: {clipiqa_value:.4f} (Placeholder - requires model)")
