# stdlib
import glob
import inspect
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Union

# third party
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
import tqdm
import wandb
import yaml  # type: ignore
from diffusers import StableDiffusion3Img2ImgPipeline
import pickle
import pandas as pd
# import cv2
from torch.utils.data import DataLoader
import torch.nn.functional as F
from piq import psnr, ssim
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image


from fmplug.utils.var_es import VarianceEarlyStopping, MeanEarlyStopping

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_NAME = "FMPlug"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """
    Function to set the seed for the run
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(123)  # Set a fixed seed for reproducibility


def resolve_repo_path(path_value: Union[str, os.PathLike[str]]) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_data_path(
    data_root: Path,
    path_value: Optional[Union[str, os.PathLike[str]]],
    default_relative_path: str,
) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(path_value or default_relative_path))))
    if not path.is_absolute():
        path = data_root / path
    return path.resolve()


def require_existing_path(path: Path, setting_name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{setting_name} points to '{path}', but that path does not exist. "
            "Update fmplug/configs/<config>.yaml or set INVERSEBENCH_DATA_ROOT."
        )
    return path


def get_torch_dtype(dtype_name: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    dtype_key = str(dtype_name).replace("torch.", "")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
    }
    if dtype_key not in dtype_map:
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'. Expected one of {sorted(dtype_map)}.")
    return dtype_map[dtype_key]


def setup_inversebench(scientific_config: dict[str, Any]) -> Path:
    inversebench_root = resolve_repo_path(
        os.environ.get("INVERSEBENCH_ROOT")
        or scientific_config.get("inversebench_path")
        or "InverseBench"
    )
    if not inversebench_root.exists():
        raise FileNotFoundError(
            "InverseBench is required for scientific tasks. Clone it from "
            "https://github.com/devzhk/InverseBench.git and set scientific.inversebench_path "
            "or INVERSEBENCH_ROOT."
        )
    inversebench_root_text = str(inversebench_root)
    if inversebench_root_text not in sys.path:
        sys.path.insert(0, inversebench_root_text)
    return inversebench_root


def import_inversebench_components() -> dict[str, Any]:
    try:
        from training.dataset import BlackHole, LMDBData, MultiCoilMRILMDBData
        from training.loss import DynamicRangePSNRLoss, DynamicRangeSSIMLoss
        import eval as inversebench_eval
    except ImportError as exc:
        raise ImportError(
            "Could not import InverseBench modules. Confirm that scientific.inversebench_path "
            "or INVERSEBENCH_ROOT points to a cloned InverseBench repository."
        ) from exc

    return {
        "BlackHole": BlackHole,
        "LMDBData": LMDBData,
        "MultiCoilMRILMDBData": MultiCoilMRILMDBData,
        "DynamicRangePSNRLoss": DynamicRangePSNRLoss,
        "DynamicRangeSSIMLoss": DynamicRangeSSIMLoss,
        "BlackHoleEvaluator": getattr(inversebench_eval, "BlackHoleEvaluator", None)
        or getattr(inversebench_eval, "BlackHole"),
        "AcousticWaveEvaluator": getattr(inversebench_eval, "AcousticWaveEvaluator", None)
        or getattr(inversebench_eval, "AcousticWave"),
        "MRIEvaluator": getattr(inversebench_eval, "MRIEvaluator", None)
        or getattr(inversebench_eval, "MRI"),
        "InverseScatterEvaluator": getattr(inversebench_eval, "InverseScatterEvaluator", None)
        or getattr(inversebench_eval, "InverseScatter"),
        "ImageEvaluator": getattr(inversebench_eval, "ImageEvaluator", None)
        or getattr(inversebench_eval, "Image"),
        "NavierStokes2dEvaluator": getattr(inversebench_eval, "NavierStokes2dEvaluator", None)
        or getattr(inversebench_eval, "NavierStokes2d", None),
    }



def configure_wandb(scientific_config: dict[str, Any]) -> str:
    wandb_mode = scientific_config.get("wandb_mode", os.environ.get("WANDB_MODE", "offline"))
    os.environ.setdefault("WANDB_MODE", str(wandb_mode))
    api_key = scientific_config.get("wandb_api_key") or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)  # type: ignore
    return str(scientific_config.get("wandb_project_prefix", PROJECT_NAME))

class Normalizer:
    def __init__(self, dataloader: torch.utils.data.DataLoader, device='cuda'):
        """
        Args:
            dataloader: DataLoader containing test data
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.global_min, self.global_max = self.compute_global_min_max(dataloader)
        print(f"Global min: {self.global_min}, Global max: {self.global_max}")

    def compute_global_min_max(self, dataloader):
        """
        Compute global min and max across the entire dataset.
        """
        min_val = float('inf')
        max_val = float('-inf')
        for batch in dataloader:
            img = batch['target']
            img = img.to(self.device)
            min_val = min(min_val, img.min().item())
            max_val = max(max_val, img.max().item())
        return min_val, max_val

    def normalize(self, img: torch.Tensor):
        """
        Normalize tensor to [-1, 1] using global min/max.
        """
        return 2 * (img - self.global_min) / (self.global_max - self.global_min) - 1

    def unnormalize(self, img_norm: torch.Tensor):
        """
        Revert normalization back to original range.
        """
        return (img_norm + 1) / 2 * (self.global_max - self.global_min) + self.global_min

def two_to_rgb_lossless(img_2ch: torch.Tensor) -> torch.Tensor:
    """
    img_2ch: [B, 2, H, W] tensor
    returns rgb: [B, 3, H, W]
    """
    B, C, H, W = img_2ch.shape
    assert C == 2

    rgb = torch.zeros((B, 3, H, W), dtype=img_2ch.dtype, device=img_2ch.device)

    # Put original channels in R and G (lossless storage)
    rgb[:, 0, :, :] = img_2ch[:, 0, :, :]  # R
    rgb[:, 1, :, :] = img_2ch[:, 1, :, :]  # G

    # Blue channel = average (for visualization only)
    # rgb[:, 2, :, :] = 0.5 * (img_2ch[:, 0, :, :] + img_2ch[:, 1, :, :])

    return rgb


def rgb_to_two_lossless(rgb: torch.Tensor) -> torch.Tensor:
    """
    rgb: [B, 3, H, W]
    returns img_2ch: [B, 2, H, W]
    """
    # Just extract R and G, since those hold original data
    return rgb[:, 0:2, :, :]


def grayscale_to_rgb(gray: torch.Tensor) -> torch.Tensor:
    """
    Lossless storage of grayscale in RGB.
    gray: [B, 1, H, W] or [1, H, W]
    returns: [B, 3, H, W] or [3, H, W]
    """
    if gray.dim() == 3:  # [1, H, W]
        gray = gray.unsqueeze(0)  # [1, 1, H, W]
        squeeze_back = True
    else:
        squeeze_back = False

    B, C, H, W = gray.shape
    assert C == 1

    rgb = torch.zeros((B, 3, H, W), dtype=gray.dtype, device=gray.device)
    rgb[:, 0, :, :] = gray[:, 0, :, :]  # store grayscale in R
    rgb[:, 1, :, :] = gray[:, 0, :, :]  # optional: copy for visualization
    rgb[:, 2, :, :] = gray[:, 0, :, :]  # optional: copy for visualization

    if squeeze_back:
        rgb = rgb.squeeze(0)
    return rgb


def rgb_to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    """
    Recover grayscale from RGB (lossless, only read R channel).
    rgb: [B, 3, H, W] or [3, H, W]
    returns: [B, 1, H, W] or [1, H, W]
    """
    if rgb.dim() == 3:  # [3, H, W]
        rgb = rgb.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    gray = rgb[:, 0:1, :, :]  # extract stored grayscale

    if squeeze_back:
        gray = gray.squeeze(0)
    return gray


def relative_l1_loss(pred, target, eps=1e-5):
    """
    Relative L1 loss that reduces the influence of high absolute values.

    Args:
        pred: Predicted image tensor, shape (N, C, H, W)
        target: Ground truth image tensor, shape (N, C, H, W)
        eps: Small constant to avoid division by zero

    Returns:
        Scalar loss
    """
    diff = torch.abs(pred - target)
    denom = torch.abs(target) + eps
    relative_error = diff / denom
    return relative_error.mean()


def hybrid_loss(pred, target, alpha=0.8):
    return alpha * torch.nn.functional.mse_loss(pred, target) + (1 - alpha) * relative_l1_loss(pred, target)


def kl_regularization(z):
    """
    Computes KL divergence between the empirical Gaussian N(mu, sigma^2)
    and the standard normal N(0, 1) for a batch of latent vectors z.
    
    Args:
        z (Tensor): shape (batch_size, latent_dim)
    
    Returns:
        kl_loss (Tensor): scalar
    """
    # Empirical mean and variance across the batch
    mu = torch.mean(z, dim=0)
    var = torch.var(z, dim=0, unbiased=False)  # use biased estimator to match VAE KL

    # KL divergence between N(mu, var) and N(0, 1)
    kl = 0.5 * torch.sum(mu**2 + var - torch.log(var + 1e-8) - 1)

    return kl


# def save_channels_png(array, path_prefix):
#     """
#     Save each channel of a (1, C, W, H) array as separate PNG images.
    
#     Args:
#         array (np.ndarray): Input array of shape (1, C, W, H)
#         path_prefix (str): Prefix path to save images (e.g., "./image")
#     """
#     # Remove batch dimension
#     array = array[0]  # now shape is (C, W, H)

#     C = array.shape[0]
#     for c in range(C):
#         channel = array[c]

#         # Normalize / clip and convert to uint8
#         channel = np.clip(channel, 0, 1)
#         channel_uint8 = (channel * 255).round().astype(np.uint8)

#         # Convert to HWC for saving as single-channel grayscale
#         channel_hwc = np.transpose(channel_uint8, (1, 0))  # W,H -> H,W

#         save_path = f"{path_prefix}_channel{c}.png"
#         cv2.imwrite(save_path, channel_hwc)
#         print(f"Saved channel {c} to {save_path}")
#     if C == 2:
#         channel = np.sqrt(array[0]**2 + array[1]**2)
#         # Normalize / clip and convert to uint8
#         channel = np.clip(channel, 0, 1)
#         channel_uint8 = (channel * 255).round().astype(np.uint8)

#         # Convert to HWC for saving as single-channel grayscale
#         channel_hwc = np.transpose(channel_uint8, (1, 0))  # W,H -> H,W

#         save_path = f"{path_prefix}_magnitude.png"
#         cv2.imwrite(save_path, channel_hwc)
#         print(f"Saved magnitude")
            

def save_channels_png(array, path_prefix):
    """
    Save each channel of a (1, C, W, H) array as separate PNG images using torchvision.
    """
    # 1. Convert to Tensor if input is numpy
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array

    # 2. Remove batch dimension: (1, C, W, H) -> (C, W, H)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    C = tensor.shape[0]

    for c in range(C):
        # Extract channel: (W, H)
        channel = tensor[c]

        # Transpose to match your original logic: (W, H) -> (H, W)
        channel = channel.permute(1, 0)
        
        # Add a channel dimension for save_image: (1, H, W)
        channel = channel.unsqueeze(0)

        save_path = f"{path_prefix}_channel{c}.png"
        
        # save_image automatically clamps values to [0, 1] for float inputs
        save_image(channel, save_path)
        print(f"Saved channel {c} to {save_path}")

    # 3. Handle Vector Field Magnitude
    if C == 2:
        # Calculate magnitude: sqrt(x^2 + y^2)
        magnitude = torch.sqrt(tensor[0]**2 + tensor[1]**2)
        
        # Transpose: (W, H) -> (H, W)
        magnitude = magnitude.permute(1, 0)
        
        # Add channel dimension: (1, H, W)
        magnitude = magnitude.unsqueeze(0)

        save_path = f"{path_prefix}_magnitude.png"
        save_image(magnitude, save_path)
        print(f"Saved magnitude")

def visualize_image(ref: np.array, measurement: np.array, output: np.array, save_file_name: str) -> None:
    # Create a figure with 1 row, 3 columns
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 10))

    # Ground truth
    ax1.imshow(ref)
    ax1.set_title("Ground Truth Image")
    ax1.axis("off")

    # Display the corrupted image
    ax2.imshow(measurement)
    ax2.set_title("Corrupted Image")
    ax2.axis("off")

    # Reconstructed image
    ax3.imshow(output)
    ax3.set_title("Reconstructed Image (FM)")
    ax3.axis("off")

    # Pixel-wise absolute difference (grayscale)
    diff = np.abs(ref - output).astype(float).mean(axis=-1)
    ax4.imshow(diff, cmap="hot")
    ax4.set_title("Pixel Difference (FM)")
    ax4.axis("off")

    # Save the figure
    plt.tight_layout()
    fig.savefig(
        os.path.join(save_file_name),
        bbox_inches="tight",
    )
    plt.close()


def gauss_sphere_reg(z, low=0.975, high=1.025):
    with torch.no_grad():
        norm = torch.norm(z, p=2)
        target = math.sqrt(z.numel())

        if norm < low * target:
            z.data = z / norm * target * low
        elif norm > high * target:
            z.data = z / norm * target * high
        else:
            z.data = z
    return z


def integrate(
    f,
    z,
    t,
    NFE,
    prompt_embedding,
    pooled_prompt_embedding,
    is_calibrate,
    device,
    guidance_scale: float = 7.0,
    method: str = "heun2"):
    do_classifier_free_guidance = guidance_scale > 1.0
    # print("t: ", t)
    # print("NFE: ", NFE)
    temp_t = 1000 * t
    # print("temp_t: ", temp_t)
    delta_t = temp_t / NFE
    temp_t_next = temp_t - delta_t 
    
    sigma = temp_t / 1000
    sigma_next = temp_t_next / 1000
    zt = z

    for i in range(NFE):
        # Rebuilt from the current zt every step, not just once before the
        # loop, so each step feeds the latent produced by the previous step.
        latent_model_input = torch.cat([zt] * 2) if do_classifier_free_guidance else zt

        time_step = temp_t.expand(latent_model_input.shape[0])
        time_step_next = temp_t_next.expand(latent_model_input.shape[0])
        if method == 'euler':
            noise_pred = f(
                x=latent_model_input,
                t=time_step,
                prompt_embedding=prompt_embedding,
                pooled_embedding=pooled_prompt_embedding,
                device=device,
            )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            else:
                noise_pred, noise_pred_text = noise_pred.chunk(2)

        elif method == 'heun2':
            dt = sigma_next - sigma
            temp_k1 = f(
                x=latent_model_input,
                t=time_step,
                prompt_embedding=prompt_embedding,
                pooled_embedding=pooled_prompt_embedding,
                device=device,
            )
            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = temp_k1.chunk(2)
                k1 = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            else:
                k1, noise_pred_text = temp_k1.chunk(2)

            # Predict next latent using Euler step
            x1_pred = latent_model_input + dt * k1

            # k2
            temp_k2 = f(
                x=x1_pred,
                t=time_step_next,
                prompt_embedding=prompt_embedding,
                pooled_embedding=pooled_prompt_embedding,
                device=device,
            )
            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = temp_k2.chunk(2)
                k2 = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            else:
                k2, noise_pred_text = temp_k2.chunk(2)
            
            # Heun2 step (average slope)
            noise_pred = 0.5 * dt * (k1 + k2)


        # Update step for huen2
        zt = zt + noise_pred
        
        temp_t = temp_t - delta_t 
        temp_t_next = temp_t - delta_t 
    
        sigma = temp_t / 1000
        sigma_next = temp_t_next / 1000
        # print("temp_t: ", temp_t)

    return zt


def solve(config_name: str) -> None:
    global device

    # Load in the configuration
    base_config_path = REPO_ROOT / "fmplug" / "configs"
    with open(base_config_path / f"{config_name}.yaml", "r") as file:
        config_all = yaml.safe_load(file)

    # Now safely access each section
    fmplug_config = config_all["fmplug"]
    scientific_config = config_all.get("scientific", {})
    project_name = fmplug_config.get("project_name", config_name)

    device_name = str(scientific_config.get("device", fmplug_config.get("device", "cuda")))
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Scientific FMPlug tasks require an available CUDA device.")
    set_seed(int(scientific_config.get("seed", 123)))

    inversebench_root = setup_inversebench(scientific_config)
    inversebench_components = import_inversebench_components()
    MultiCoilMRILMDBData = inversebench_components["MultiCoilMRILMDBData"]
    LMDBData = inversebench_components["LMDBData"]
    BlackHole = inversebench_components["BlackHole"]
    DynamicRangePSNRLoss = inversebench_components["DynamicRangePSNRLoss"]
    DynamicRangeSSIMLoss = inversebench_components["DynamicRangeSSIMLoss"]
    BlackHoleEvaluator = inversebench_components["BlackHoleEvaluator"]
    AcousticWaveEvaluator = inversebench_components["AcousticWaveEvaluator"]
    MRIEvaluator = inversebench_components["MRIEvaluator"]
    InverseScatterEvaluator = inversebench_components["InverseScatterEvaluator"]
    ImageEvaluator = inversebench_components["ImageEvaluator"]
    NavierStokes2dEvaluator = inversebench_components["NavierStokes2dEvaluator"]

    dr_psnr_loss = DynamicRangePSNRLoss()
    dr_ssim_loss = DynamicRangeSSIMLoss()
    norm_checkpoint = resolve_repo_path(scientific_config.get("norm_checkpoint", "best_meanvar_model.pth"))
    project_prefix = configure_wandb(scientific_config)

    if device.type == "cuda":
        print("Init Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
        print("Init Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**2, 2), "MB")
    
    method = fmplug_config["method"]
    save_folder = fmplug_config["save_folder"]
    image_size = fmplug_config["img_size"]
    alpha = fmplug_config["alpha"]
    NFE = fmplug_config["NFE"]
    guidance_scale = fmplug_config["guidance_scale"]
    lr_z = fmplug_config["lr_z"]
    lr_alpha_ada = fmplug_config["lr_alpha_ada"]
    lr_prior_weight = fmplug_config.get("lr_prior_weight", lr_alpha_ada)
    num_instances = fmplug_config.get("num_instances", -1)
    decay_factor =  fmplug_config["decay_factor"]
    epochs = fmplug_config["epochs"]
    loss_multiplier = fmplug_config["loss_multiplier"]
    loss_fn = fmplug_config["loss_fn"]
    is_gauss_reg = fmplug_config["is_gauss_reg"]
    is_calibrate = fmplug_config["is_calibrate"]
    t_end = fmplug_config["t_end"]
    data_type = get_torch_dtype(fmplug_config.get("data_type", "torch.bfloat16"))
    optimizer_select = fmplug_config["optimizer_select"]
    
    lr_dec = fmplug_config["lr_dec"]
    finetune_decoder_blocks_interval = fmplug_config.get("finetune_decoder_blocks_interval", 0)
    
    es_window_size = fmplug_config["es_window_size"]
    es_patience = fmplug_config["es_patience"]
    es_min_epochs  = fmplug_config["es_min_epochs"]
    es_delta  = fmplug_config["es_delta"]
    
    
    measure_config = config_all['measurement']
    task = measure_config["operator"]["name"]
    batch_size = int(fmplug_config.get("batch_size", 1))
    data_root = resolve_repo_path(
        os.environ.get("INVERSEBENCH_DATA_ROOT")
        or scientific_config.get("data_root")
        or inversebench_root
    )
    datasets_config = scientific_config.get("datasets", {})

    def dataset_path(key: str, default_relative_path: str) -> str:
        path = resolve_data_path(data_root, datasets_config.get(key), default_relative_path)
        return str(require_existing_path(path, f"scientific.datasets.{key}"))

    if "MRI" in task:
        from inverse_problems.multi_coil_mri import MultiCoilMRI
        mri_config = scientific_config.get("mri", {})
        mri_operator_config = {
            "sigma_noise": 0.0,
            "total_lines": 320,
            "acceleration_ratio": 4,
            "pattern": "random",
            "mask_seed": 0,
        }
        mri_operator_config.update(mri_config.get("operator", {}))
        forward_op = MultiCoilMRI(**mri_operator_config)
        
        testset = MultiCoilMRILMDBData(
            root=dataset_path("mri_test", "knee_test_lmdb"),
            image_size=mri_config.get("image_size", [320, 320]),
            id_list=mri_config.get("test_id_list"),
            simulated_kspace=mri_config.get("simulated_kspace", False),
        )
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        
        
        valset = MultiCoilMRILMDBData(
            root=dataset_path("mri_val", "knee_val_lmdb"),
            image_size=mri_config.get("image_size", [320, 320]),
            id_list=mri_config.get("val_id_list"),
            simulated_kspace=mri_config.get("simulated_kspace", False),
        )
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        
    elif "scatter" in task:
        from inverse_problems.inverse_scatter import InverseScatter
        scatter_config = scientific_config.get("scatter", {})
        scatter_operator_config = {
            "Lx": 0.18,
            "Ly": 0.18,
            "Nx": 128,
            "Ny": 128,
            "wave": 6,
            "numRec": 360,
            "numTrans": 20,
            "sensorRadius": 1.6,
            "sigma_noise": 0.0001,
            "unnorm_shift": 1.0,
            "unnorm_scale": 0.5,
        }
        scatter_operator_config.update(scatter_config.get("operator", {}))
        forward_op = InverseScatter(**scatter_operator_config)
        
        testset = LMDBData(
            root=dataset_path("scatter_test", "inv-scatter-test"),
            resolution=scatter_config.get("resolution", 128),
            std=scatter_config.get("std", 0.25),
            mean=scatter_config.get("mean", 0.5),
            id_list=scatter_config.get("test_id_list", "0-99"),
        )
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        
        
        valset = LMDBData(
            root=dataset_path("scatter_val", "inv-scatter-val"),
            resolution=scatter_config.get("resolution", 128),
            std=scatter_config.get("std", 0.25),
            mean=scatter_config.get("mean", 0.5),
            id_list=scatter_config.get("val_id_list", "0-9"),
        )
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        
        # def scatter_sd_norm(x):
        #     return x * 2.0 - 1.0
        
        # def scatter_sd_unnorm(x):
        #     return (x + 1.0) / 2.0
    
    elif "fwi" in task:
        from inverse_problems.acoustic import AcousticWave
        print("FWI data loading ...")
        fwi_config = scientific_config.get("fwi", {})
        fwi_operator_config = {
            "shape": [128, 128],
            "spacing": [20.0, 10.0],
            "tn": 1000.0,
            "f0": 0.005,
            "dt": 1.0,
            "nbl": 80,
            "nshots": 16,
            "nreceivers": 129,
            "unnorm_scale": 1.0,
            "unnorm_shift": 3.0,
            "src_depth": 1270.0,
        }
        fwi_operator_config.update(fwi_config.get("operator", {}))
        forward_op = AcousticWave(**fwi_operator_config)
        
        testset = LMDBData(
            root=dataset_path("fwi_test", "fwi-test"),
            resolution=fwi_config.get("resolution", 128),
            raw_resolution=fwi_config.get("raw_resolution", 70),
            std=fwi_config.get("std", 500.0),
            mean=fwi_config.get("mean", 5000.0),
            id_list=fwi_config.get("test_id_list", "0-99"),
        )
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        
        
        valset = LMDBData(
            root=dataset_path("fwi_val", "fwi-val"),
            resolution=fwi_config.get("resolution", 128),
            raw_resolution=fwi_config.get("raw_resolution", 70),
            std=fwi_config.get("std", 500.0),
            mean=fwi_config.get("mean", 5000.0),
            id_list=fwi_config.get("val_id_list", "1-10"),
        )
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        
        def fwi_norm(x):
            return (x - 1.5) / 3.0
        
        def fwi_sd_norm(x):
            return (x - 1.5) / 3.0
        
        def fwi_sd_unnorm(x):
            return x * 3.0 + 1.5
        
        
    elif "blackhole" in task:
        from inverse_problems.blackhole import BlackHoleImaging
        blackhole_config = scientific_config.get("blackhole", {})
        blackhole_operator_config = {
            "root": dataset_path("blackhole_measure", "blackhole/measure"),
            "imsize": 64,
            "observation_time_ratio": 1.0,
            "noise_type": "eht",
            "w1": 0,
            "w2": 1,
            "w3": 1,
            "w4": 0.5,
            "sigma_noise": 0.0,
            "unnorm_scale": 0.5,
            "unnorm_shift": 1.0,
        }
        blackhole_operator_config.update(blackhole_config.get("operator", {}))
        forward_op = BlackHoleImaging(**blackhole_operator_config)
        
        testset = BlackHole(
            root=dataset_path("blackhole_test", "blackhole/test"),
            resolution=blackhole_config.get("resolution", 64),
            original_resolution=blackhole_config.get("original_resolution", 64),
            random_flip=blackhole_config.get("random_flip", False),
            zoom_in_out=blackhole_config.get("zoom_in_out", False),
            id_list=blackhole_config.get("test_id_list"),
        )
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        
        
        valset = BlackHole(
            root=dataset_path("blackhole_val", "blackhole/valid"),
            resolution=blackhole_config.get("resolution", 64),
            original_resolution=blackhole_config.get("original_resolution", 64),
            random_flip=blackhole_config.get("random_flip", False),
            zoom_in_out=blackhole_config.get("zoom_in_out", False),
            id_list=blackhole_config.get("val_id_list"),
        )
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

        
    else:
        raise ValueError(f"Unsupported scientific task '{task}'. Expected MRI, scatter, fwi, or blackhole.")

    if num_instances == -1:
        num_instances = len(valloader)
    normalizer_test = Normalizer(testloader)
    normalizer = Normalizer(valloader)
    # print("stop: ", stop)

        
    # prior_images = 0
    # for prior_image in valloader:
    #     prior_images += prior_image['target']
    #     # prior_images = prior_image['target']
    # prior_image = prior_images / len(valloader)
    # # prior_image = prior_images
    
    wandb_instance = wandb.init(  # type: ignore
            # set the wandb project where this run will be logged
            project=project_prefix+"-"+project_name,
            tags=["Experimental", task],
            config={
                "method": method,
                "optimizer": optimizer_select,
                "lr_z": lr_z,
                "lr_alpha_ada": lr_alpha_ada,
                "lr_dec": lr_dec,
                "decay_factor": decay_factor,
                "epochs": epochs,
                "loss_multiplier": loss_multiplier,
                "image_size": image_size,
                "NFE": NFE,
                "guidance_scale": guidance_scale,
                "loss_fn": loss_fn,
                "t_end": t_end,
                "data_type": data_type,
            },
        )
    
    save_dir = os.path.join(save_folder, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    for i, data in enumerate(testloader):
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, dict):
            assert 'target' in data.keys(), "'target' must be in the data dict"
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    data[key] = val.to(device)
        data_id = testset.id_list[i]

        measurement = forward_op(data)
        gt_img = data['target']
        print("gt_img: ", gt_img)
        print("gt_img: ", gt_img.min())
        print("gt_img: ", gt_img.max())
                
        # print("gt_img: ", gt_img.shape)
        # print("gt_img: ", gt_img.min(), gt_img.min())
        # print("measurement: ", measurement.shape)
        # print("measurement: ", measurement)
        # print("measurement: ", measurement.min(), measurement.max())
        # print("data: ",data.keys())
        

        ref_numpy = np.array(gt_img.cpu().numpy())

        ref_img = torch.Tensor(ref_numpy).to(data_type)
        ref_img.requires_grad = False

        print("Load in SD3 image to image model ...")
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            text_encoder_3=None,
            tokenizer_3=None,
            dtype=data_type,
        )
        pipe = pipe.to(device, dtype=data_type)
        if pipe.text_encoder is not None:
            pipe.text_encoder.to(dtype=torch.float32)
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.to(dtype=torch.float32)
        # pipe.enable_model_cpu_offload()

        # Extract different components of the pipeline
        prompt_encoder = pipe.encode_prompt

        # Transformer block
        transformer = pipe.transformer
        transformer.eval()
        transformer.requires_grad_(False)
        transformer.enable_gradient_checkpointing()

        # VAE
        vae = pipe.vae
        vae.eval()
        vae.encoder.requires_grad_(False)
        vae.decoder.requires_grad_(False) # Freeze decoder initially
        vae.enable_gradient_checkpointing()

        prompt_2 = None
        prompt_3 = None

        negative_prompt = ""
        negative_prompt_2 = None
        negative_prompt_3 = None

        do_classifier_free_guidance = True
        prompt_embeds = None
        negative_prompt_embeds = None
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        clip_skip = None
        num_images_per_prompt = 1
        max_sequence_length = 256
        lora_scale = None

        # encode prompt
        print("Encode prompt ...")
        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = prompt_encoder(
                prompt="",
                prompt_2=prompt_2,
                prompt_3=prompt_3,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                do_classifier_free_guidance=do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                clip_skip=clip_skip,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # prompt embeds with classifier free guidance
        prompt_embedding = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_embedding = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        
        del pipe
        del prompt_encoder
        torch.cuda.empty_cache()
        
        # Current memory usage by tensors (in MB)
        print("Model Load Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
        print("Model Load Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**2, 2), "MB")
        
        def encode(image: torch.Tensor) -> torch.Tensor:
            z = vae.encode(image).latent_dist.sample()
            z = (z-vae.config.shift_factor) * vae.config.scaling_factor
            return z

        def decode(z: torch.Tensor) -> torch.Tensor:
            z = (z/vae.config.scaling_factor) + vae.config.shift_factor
            return vae.decode(z, return_dict=False)[0]
        
        # if optimizer_select == "adam":
        early_stop_indicator = VarianceEarlyStopping(es_window_size, es_patience, es_min_epochs, es_delta)

        # initialize latent
        
        # measurement = measurement.detach().to(device).to(data_type)
        # measurement_numpy = measurement.squeeze(0).detach().cpu().to(torch.float32).numpy()
        # measurement.requires_grad = False
        if "MRI" in task:
            rel_dir = f"MRI/{i}"
            os.makedirs(os.path.join(save_dir, rel_dir), exist_ok=True)
            with torch.amp.autocast("cuda", dtype=data_type):
                instance_prior = []
                for i, prior_image in enumerate(valloader):
                    prior_image = prior_image['target']
                    target_size = prior_image.shape[-1]
                    if num_instances == len(instance_prior):
                        break
                    with torch.no_grad():
                        np.save(os.path.join(save_dir, rel_dir, f"prior_{len(instance_prior)-1}.npy"), prior_image)
                        prior_image = F.interpolate(prior_image, size=image_size)
                        prior_image_rgb = two_to_rgb_lossless(normalizer.normalize(prior_image))
                        # print("prior_image_rgb: ", prior_image_rgb.shape)
                        prior_image_rgb = prior_image_rgb.to(device)
                        latent = encode(prior_image_rgb)
                        instance_prior.append(latent)
                        save_image(prior_image_rgb, os.path.join(save_dir, rel_dir, f"prior_{len(instance_prior)-1}.png"), normalize=True, value_range=(-1, 1))
           
        elif "scatter" in task:
            rel_dir = f"Scatter/{i}"
            os.makedirs(os.path.join(save_dir, rel_dir), exist_ok=True)
            with torch.amp.autocast("cuda", dtype=data_type):
                instance_prior = []
                for i, prior_image in enumerate(valloader):
                    prior_image = prior_image['target']
                    target_size = prior_image.shape[-1]
                    if num_instances == len(instance_prior):
                        break
                    with torch.no_grad():
                        np.save(os.path.join(save_dir, rel_dir, f"prior_{len(instance_prior)-1}.npy"), prior_image)
                        prior_image = F.interpolate(prior_image, size=image_size)
                        prior_image_rgb = grayscale_to_rgb(prior_image)
                        # prior_image_rgb = grayscale_to_rgb(normalizer.normalize(prior_image))
                        # prior_image_rgb = grayscale_to_rgb(scatter_sd_unnorm(prior_image))
                        # print("prior_image_rgb: ", prior_image_rgb.shape)
                        
                        prior_image_rgb = prior_image_rgb.to(device)
                        latent = encode(prior_image_rgb)
                        instance_prior.append(latent)
                        save_image(prior_image_rgb, os.path.join(save_dir, rel_dir, f"prior_{len(instance_prior)-1}.png"), normalize=True, value_range=(-1, 1))
                        
        elif "fwi" in task:
            print("FWI prior loading ...")
            rel_dir = f"FWI/{i}"
            os.makedirs(os.path.join(save_dir, rel_dir), exist_ok=True)
            with torch.amp.autocast("cuda", dtype=data_type):
                instance_prior = []
                for i, prior_image in enumerate(valloader):
                    prior_image = prior_image['target']
                    target_size = prior_image.shape[-1]
                    if num_instances == len(instance_prior):
                        break
                    with torch.no_grad():
                        np.save(os.path.join(save_dir, rel_dir, f"prior_{len(instance_prior)-1}.npy"), prior_image)
                        prior_image = F.interpolate(prior_image, size=image_size)
                        # Move channels to front → (2, 1, W, H)
                        # prior_image = normalizer.normalize(prior_image)
                        prior_image = fwi_sd_norm(prior_image)
                        prior_image_rgb = grayscale_to_rgb(prior_image)
                        # Repeat → (2, 3, W, H)
                        # prior_image_rgb = prior_image.repeat(1, 3, 1, 1)
                        # print("prior_image_rgb: ", prior_image_rgb.shape)
                        prior_image_rgb = prior_image_rgb.to(device)
                        latent = encode(prior_image_rgb)
                        # print("latent: ", latent.shape)
                        
                        instance_prior.append(latent)
        elif "blackhole" in task:
            rel_dir = f"blackhole/{i}"
            os.makedirs(os.path.join(save_dir, rel_dir), exist_ok=True)
            with torch.amp.autocast("cuda", dtype=data_type):
                instance_prior = []
                for i, prior_image in enumerate(valloader):
                    prior_image = prior_image['target']
                    target_size = prior_image.shape[-1]
                    if num_instances == len(instance_prior):
                        break
                    with torch.no_grad():
                        np.save(os.path.join(save_dir, rel_dir, f"prior_{len(instance_prior)-1}.npy"), prior_image)
                        prior_image = F.interpolate(prior_image, size=image_size)
                        prior_image_rgb = grayscale_to_rgb(prior_image)
                        # prior_image_rgb = grayscale_to_rgb(normalizer.normalize(prior_image))
                        # prior_image_rgb = grayscale_to_rgb(scatter_sd_unnorm(prior_image))
                        # print("prior_image_rgb: ", prior_image_rgb.shape)
                        
                        prior_image_rgb = prior_image_rgb.to(device=device, dtype=data_type)
                        latent = encode(prior_image_rgb)
                        instance_prior.append(latent)
                        save_image(prior_image_rgb, os.path.join(save_dir, rel_dir, f"prior_{len(instance_prior)-1}.png"), normalize=True, value_range=(-1, 1))
        
        print("num of instances: ", len(instance_prior))                     
        if len(instance_prior) > 0:
            instance_prior = torch.cat(instance_prior, dim=0)
            weighted_prior = torch.nn.Parameter(torch.ones(len(instance_prior), device=device))

            instance_prior = instance_prior.detach()
            instance_prior = instance_prior.requires_grad_(False)
        else:
            latent = encode(torch.ones([1, 3, image_size, image_size], device=device))
            alpha = -6.
            weighted_prior = torch.nn.Parameter(torch.ones(1, device=device))
            instance_prior = torch.zeros_like(latent).detach()


            
        z = torch.randn_like(latent)
        z = torch.nn.parameter.Parameter(z, True).to(device)
        z = z.requires_grad_(True)

        
        alpha_ada = torch.tensor(alpha).to(device)
        alpha_ada = alpha_ada.requires_grad_(True)
        t_ada = (1 - torch.sigmoid((alpha_ada-0.5)*6))
        
        scale_param = nn.Parameter(torch.tensor(1.0, device=z.device)).requires_grad_(True)
        shift_param = nn.Parameter(torch.tensor(0.0, device=z.device)).requires_grad_(True)
        
        # Solve the ODE
        print("Solve inverse problem ...")

        def f(x, t, prompt_embedding, pooled_embedding, device):
            with torch.amp.autocast(device.type, dtype=data_type):
                result = transformer(
                    hidden_states=x,
                    timestep=t,
                    encoder_hidden_states=prompt_embedding,
                    pooled_projections=pooled_embedding,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            return result

        def checkpointed_integrate(z, t_ada):
            return integrate(
                f,
                z,
                t_ada,
                NFE,
                prompt_embedding,
                pooled_embedding,
                is_calibrate, device,
                guidance_scale=guidance_scale,
                method=method
            )

        # Criterion for learning
        if loss_fn == "l1":
            criterion = torch.nn.L1Loss().to(device)
        elif loss_fn == "mse":
            criterion = torch.nn.MSELoss().to(device)
        elif loss_fn == "l1_rel":
            criterion = relative_l1_loss
        elif loss_fn == "hybrid":
            criterion = hybrid_loss
        
        if optimizer_select == "adam":
            params_group1 = {'params': z, 'lr': lr_z[0]}
            params_group2 = {'params': alpha_ada, 'lr': lr_alpha_ada}
            params_group3 = {'params': weighted_prior, 'lr': lr_prior_weight}
            # params_group4 = {'params': scale_param, 'lr': lr_shift}
            # params_group5 = {'params': shift_param, 'lr': lr_shift}
            if len(instance_prior) > 1:
                optimizer = torch.optim.AdamW([params_group1, params_group2, params_group3])
            else:
                optimizer = torch.optim.AdamW([params_group1, params_group2])


            # Freeze all decoder parameters initially
            for p in vae.decoder.parameters():
                p.requires_grad_(False)
            

        elif optimizer_select == "lbfgs":
            optimizer = torch.optim.LBFGS([z, alpha_ada], lr_z[0])

        
        # Get decoder blocks for incremental fine-tuning
        decoder_blocks = list(vae.decoder.up_blocks) # Assuming decoder blocks are in vae.decoder.up_blocks
        num_decoder_blocks = len(decoder_blocks)
            
        psnrs = []
        data_misfits = []
        losses = []
        z_grad_norms = []
        z_deltas = []
        z_rel_updates = []
        z_prev = z.clone().detach()
        rel_update, delta, grad_norm = 0.0, 0.0, 0.0
        
        # Current memory usage by tensors (in MB)
        print("Init Opt Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
        print("Init Opt Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**2, 2), "MB")
        decoded_output = None
        decoded_output_rgb = None
        d = math.sqrt(z.numel())



        for iterator in tqdm.tqdm(range(epochs)):
            
            
            z = gauss_sphere_reg(z) if is_gauss_reg else z # TODO: randomness for stability
            # net_input = gauss_sphere_reg(z + 0.03 * torch.randn_like(z).detach()) if is_gauss_reg else z + 0.03 * torch.randn_like(z).detach()
            iter_start_time = time.time()
            
            # # Incremental fine-tuning of VAE decoder blocks
            # # if finetune_decoder_blocks_interval > 0 and iterator % finetune_decoder_blocks_interval == 0 and iterator < finetune_decoder_blocks_interval * num_decoder_blocks:
            # if finetune_decoder_blocks_interval > 0 and iterator > 0 and iterator % finetune_decoder_blocks_interval == 0 and iterator < finetune_decoder_blocks_interval * len(lr_dec):
            #     block_to_unfreeze_idx = iterator // finetune_decoder_blocks_interval - 1
            #     if block_to_unfreeze_idx < num_decoder_blocks:
            #         if lr_dec[block_to_unfreeze_idx] != 0.0:
            #             for param in decoder_blocks[block_to_unfreeze_idx].resnets.parameters():
            #                 param.requires_grad_(True)
            #             print(f"Unfreezing decoder block {block_to_unfreeze_idx} to optimizer with learning rate {lr_dec[block_to_unfreeze_idx]}")
            #             optimizer.add_param_group({'params': decoder_blocks[block_to_unfreeze_idx].resnets.parameters(), 'lr': lr_dec[block_to_unfreeze_idx]})
                        
            #         # optimizer.param_groups[0]['lr'] = lr[block_to_unfreeze_idx]
            #         optimizer.param_groups[0]['lr'] = lr_z[block_to_unfreeze_idx+1]
            #         if not is_calibrate:
            #             optimizer.param_groups[2]['lr'] = lr_z[block_to_unfreeze_idx+1]/1E3
                        
            #         # for param_group_idx in range(1, block_to_unfreeze_idx + 1):
            #         #     # Update the learning rate for previously added decoder blocks
            #         #     if optimizer.param_groups[-param_group_idx]['lr'] != 0.0:
            #         #         optimizer.param_groups[-param_group_idx]['lr'] = lr_dec[block_to_unfreeze_idx]
            #     # if block_to_unfreeze_idx == 0:
            #     #     for param in decoder_blocks[-1].parameters():
            #     #         param.requires_grad_(True)
            #     #     optimizer.add_param_group({'params': decoder_blocks[-1].parameters(), 'lr': lr_dec[block_to_unfreeze_idx]})
            #     # else:
            #     #     optimizer.param_groups[-1]['lr'] = lr_dec[block_to_unfreeze_idx]
            
            
            # Incremental fine-tuning of VAE decoder blocks
            # if finetune_decoder_blocks_interval > 0 and iterator % finetune_decoder_blocks_interval == 0 and iterator < finetune_decoder_blocks_interval * num_decoder_blocks:
            if finetune_decoder_blocks_interval > 0 and iterator > 0 and iterator % finetune_decoder_blocks_interval == 0 and iterator < finetune_decoder_blocks_interval * len(lr_dec):
                block_to_unfreeze_idx = iterator // finetune_decoder_blocks_interval - 1
                if block_to_unfreeze_idx == 0:
                    # --- STEP 1: SELECTIVE UNFREEZING ---
                    # Freeze everything first
                    for param in vae.decoder.parameters():
                        param.requires_grad = False
                        
                    trainable_params = []
                    
                    # A. Always tune Normalization (Style/Contrast)
                    for module in vae.decoder.modules():
                        if isinstance(module, nn.GroupNorm):
                            for param in module.parameters():
                                param.requires_grad = True
                                trainable_params.append(param)

                    # B. CRITICAL: Tune the Output Convolution (Feature -> Pixel mapping)
                    # This bridges the gap between RGB priors and Medical Grayscale data
                    vae.decoder.conv_out.weight.requires_grad = True
                    vae.decoder.conv_out.bias.requires_grad = True
                    trainable_params.append(vae.decoder.conv_out.weight)
                    trainable_params.append(vae.decoder.conv_out.bias)
                    
                    # C. OPTIONAL: Tune Residual Shortcuts (Geometry adjustments)
                    # These 1x1 convs allow the model to skip/alter features in the upsampling blocks
                    for name, module in vae.decoder.named_modules():
                        if 'conv_shortcut' in name:
                            module.weight.requires_grad = True
                            trainable_params.append(module.weight)
                            # bias might not exist depending on implementation, check first
                            if hasattr(module, 'bias') and module.bias is not None:
                                module.bias.requires_grad = True
                                trainable_params.append(module.bias)
                    optimizer.add_param_group({'params': trainable_params, 'lr': lr_dec[block_to_unfreeze_idx]})
                else:
                    optimizer.param_groups[-1]['lr'] = lr_dec[block_to_unfreeze_idx]
                    
                vae.decoder.train()                    
            
                        
            def closure():
                nonlocal decoded_output
                # nonlocal decoded_output_rgb
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", dtype=data_type):
                    temp_alpha = torch.sigmoid((alpha_ada - 0.5) * 6)
                    
                    weights = F.softmax(weighted_prior, dim=0)
                    # print("weights: ", weights)
                    # weights = torch.abs(weighted_prior)
                    # weights = weighted_prior / (weighted_prior.norm(p=2) + 1e-8)
                    # weights = torch.sin(weighted_prior)
                    # weights = (torch.sin(weighted_prior) + 1.) / 2.
                    weights = weights.view(-1, 1, 1, 1)
                    latent_y = torch.sum(weights * instance_prior, dim=0, keepdim=True)

                    temp_z = (1 - temp_alpha) * z + temp_alpha * latent_y

                    x_t = checkpointed_integrate(temp_z, t_ada)
                    decoded_output_rgb = decode(x_t)
                    decoded_output_rgb = F.interpolate(decoded_output_rgb, size=target_size)

                    # --- BRANCH 1: MRI -------------------------------------------------------
                    if "MRI" in task:
                        # decoded_output = normalizer.unnormalize(rgb_to_two_lossless(decoded_output_rgb)) * scale_param + shift_param
                        decoded_output = torch.sin(rgb_to_two_lossless(decoded_output_rgb)) * scale_param + shift_param
                        operator_decoded_output = forward_op.forward(decoded_output)
                        
                        loss = criterion(operator_decoded_output, measurement.to(operator_decoded_output.dtype)) * loss_multiplier
                        # loss = forward_op.loss(decoded_output, measurement.to(operator_decoded_output.dtype)) * loss_multiplier

                        # standard autograd case
                        loss.backward(retain_graph=True)
                        return loss

                    # --- BRANCH 2: scatter ----------------------------------------------------
                    elif "scatter" in task:
                        # decoded_output = normalizer.unnormalize(rgb_to_grayscale(decoded_output_rgb)) * scale_param + shift_param
                        # decoded_output = scatter_sd_unnorm(rgb_to_grayscale(decoded_output_rgb))
                        decoded_output = rgb_to_grayscale(decoded_output_rgb)
                        print("decoded_output: ", decoded_output.min(), decoded_output.max())
                        # decoded_output = torch.sin(rgb_to_grayscale(decoded_output_rgb)) * scale_param + shift_param
                        # print("decoded_output: ", decoded_output.shape)
                        # print("measurement: ", measurement.shape)
                        operator_decoded_output = forward_op.forward(decoded_output)
                        loss = forward_op.loss(decoded_output, measurement.to(operator_decoded_output.dtype)) * loss_multiplier
                        # print("loss: ", loss.item)
                        # loss = criterion(operator_decoded_output, measurement.to(operator_decoded_output.dtype)) * loss_multiplier
                        

                        # standard autograd case
                        loss.backward(retain_graph=True)
                        return loss

                    # --- BRANCH 3: fwi --------------------------------------------------------
                    elif "fwi" in task:
                        print("FWI optimization step ...")
                        
                        # decoded_output = normalizer.unnormalize(rgb_to_grayscale(decoded_output_rgb))
                        # decoded_output = fwi_sd_unnorm(rgb_to_grayscale(decoded_output_rgb))
                        decoded_output = rgb_to_grayscale(decoded_output_rgb)
                        
                        # decoded_output = rgb_to_grayscale(decoded_output_rgb) * scale_param + shift_param

                        # compute scalar loss (for reporting)
                        # loss = forward_op.loss(decoded_output, measurement) * loss_multiplier
                        # print("loss: ", loss.item())
                        print("scale_param: ", scale_param.item())
                        print("shift_param: ", shift_param.item())
                        # compute external gradient
                        gradient, loss_scale = forward_op.gradient(
                            decoded_output, measurement, return_loss=True
                        )
                        gradient = gradient.to(decoded_output.dtype).to(decoded_output.device)
                        print("loss_scale: ", loss_scale.item())
                        print("gradient: ", gradient.mean())
                        print("decoded_output: ", decoded_output.min(), decoded_output.max())
                        print("gt_img: ", gt_img.min(), gt_img.max())
                        # propagate external grad back to z
                        decoded_output.backward(gradient, retain_graph=True)

                        return loss_scale

                    # --- BRANCH 4: Blackhole --------------------------------------------------------
                    elif "blackhole" in task:
                        # decoded_output = rgb_to_grayscale(decoded_output_rgb)
                        decoded_output = rgb_to_grayscale(torch.clamp(decoded_output_rgb, min=-1.0, max=1.0))
                        print("decoded_output: ", decoded_output.min().item(), decoded_output.max().item(), decoded_output.mean().item())
                        # decoded_output = torch.sin(rgb_to_grayscale(decoded_output_rgb)) * scale_param + shift_param
                        # print("decoded_output: ", decoded_output.shape)
                        # print("measurement: ", measurement.shape)
                        # loss = forward_op.loss(decoded_output, measurement.to(operator_decoded_output.dtype)) * loss_multiplier
                        operator_decoded_output = forward_op.forward(decoded_output)
                        loss = criterion(operator_decoded_output, measurement.to(operator_decoded_output.dtype)) * loss_multiplier
                        
                        # print("loss: ", loss.item)
                        # loss = criterion(operator_decoded_output, measurement.to(operator_decoded_output.dtype)) * loss_multiplier
                        

                        # standard autograd case
                        loss.backward(retain_graph=True)
                        return loss

                    # --------------------------------------------------------------------------
                    # If none matched (should not happen)
                    else:
                        raise RuntimeError(f"Unknown task type: {task}")
            
            
            if optimizer_select == "adam":
                loss = optimizer.step(closure)
                new_lr = lr_alpha_ada * (decay_factor ** iterator)
                optimizer.param_groups[1]['lr'] = new_lr
            else:
                loss = optimizer.step(closure)
                
            t_ada = (1 - torch.sigmoid((alpha_ada-0.5)*6))
            
            grad_norm = z.grad.norm().item()
            delta = (z - z_prev).norm().item()
            rel_update = delta / (z_prev.norm() + 1e-8)

            z_grad_norms.append(grad_norm)
            z_deltas.append(delta)
            z_rel_updates.append(rel_update.item())
            z_prev = z.clone().detach()
            # scheduler.step()
            
            losses.append(loss.item())
            
            # print("Opt Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
            # print("Opt Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**2, 2), "MB")
            torch.cuda.empty_cache()
            iter_time = time.time() - iter_start_time
            # print(f"Iter Time = {iter_time:.4f} sec")
            
           # Evaluate
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=data_type):
                    # Obtain unnormalized output (CPU, Detached)
                    output = decoded_output.cpu().detach()
                
                output_numpy = decoded_output.cpu().detach().numpy()
                
                # ---------------------------------------------------------
                # 1. Pre-process inputs for Evaluator
                # ---------------------------------------------------------
                # Some tasks in the original script required shifting data from [-1, 1] to [0, 1]
                # before metric calculation. We prepare separate tensors for evaluation.
                if "MRI" in task:
                    evaluator = MRIEvaluator(forward_op=forward_op)
                elif "scatter" in task:
                    evaluator = InverseScatterEvaluator(forward_op=forward_op)
                elif "fwi" in task:
                    evaluator = AcousticWaveEvaluator(forward_op=forward_op)
                elif "blackhole" in task:
                    evaluator = BlackHoleEvaluator(forward_op=forward_op)
                elif "navier" in task:  # Added based on the classes you provided
                    evaluator = NavierStokes2dEvaluator(forward_op=forward_op)
                else:
                    # Fallback default (standard Image metrics like PSNR/SSIM/LPIPS)
                    print(f"Warning: No specific evaluator matched task '{task}'. Defaulting to Image evaluator.")
                    evaluator = ImageEvaluator(forward_op=forward_op)
                evaluator.device = device
                
                eval_pred = output.clone()
                eval_target = ref_img.clone() # ref_img is usually on CPU by this point in many pipelines, if not, .cpu() it.

                eval_pred = forward_op.unnormalize(eval_pred)
                eval_target = forward_op.unnormalize(eval_target)
                
                # print("eval_pred: ", eval_pred.min().item(), eval_pred.max().item())
                # print("eval_target: ", eval_target.min().item(), eval_target.max().item())


                # ---------------------------------------------------------
                # 2. Run Evaluator
                # ---------------------------------------------------------
                # The evaluator handles device movement internally (e.g., .to(self.device))
                # measurement is passed for data_misfit or chi-squared calculations
                if "fwi" in task:
                    metrics = evaluator(eval_pred.float(), eval_target.float(), observation=measurement)
                else:
                    metrics = evaluator(eval_pred.float(), eval_target.float(), observation=measurement.float())

                # ---------------------------------------------------------
                # 3. Standardize Data Misfit extraction
                # ---------------------------------------------------------
                # Different evaluators return misfit differently or not at all. 
                # We ensure 'data_misfit' is available for the logic that follows.
                if "data misfit" in metrics:
                    data_misfit = metrics["data misfit"]
                elif "MRI" in task:
                    # Fallback if MRI evaluator didn't return it (though the provided class does)
                    # Note: Original MRI script used forward(decoded_output) - measurement
                    # The MRI Evaluator uses forward(pred) - observation. 
                    # We stick to the Evaluator's return if present, otherwise calculate manually
                    data_misfit = metrics.get('data misfit', 
                                            torch.linalg.norm(forward_op.forward(decoded_output) - measurement).item())
                else:
                    # Fallback for tasks where Evaluator might not return generic 'data misfit' 
                    # (e.g., Blackhole returns chi2, Scatter returns PSNR/SSIM only)
                    # Original script logic: sqrt(loss)
                    data_misfit = torch.sqrt(forward_op.loss(decoded_output, measurement)).mean().item()

                # ---------------------------------------------------------
                # 4. Prepare Logging Dictionary
                # ---------------------------------------------------------
                metrics_to_log = {
                    "epoch": iterator,
                    "t_ada": 1000 * (1 - torch.sigmoid((alpha_ada-0.5)*6)).item(),
                    "alpha_ada": (torch.sigmoid((alpha_ada-0.5)*6)).item(),
                    "loss": loss.item()/loss_multiplier,
                    "data_misfit": data_misfit,
                    "min_weight": F.softmax(weighted_prior, dim=0).min().item(),
                    "max_weight": F.softmax(weighted_prior, dim=0).max().item(),
                    "weights": F.softmax(weighted_prior, dim=0).tolist(),
                    "z_rel_updates": rel_update.item(),
                    "iter_time": iter_time,
                }

                # Merge the evaluator's specific metrics (PSNR, SSIM, Chi2, etc.) into the log dict
                metrics_to_log.update(metrics)

                # ---------------------------------------------------------
                # 5. Task-Specific Visualization (Blackhole)
                # ---------------------------------------------------------
                if "blackhole" in task:
                    log_images = (iterator % 50 == 0)
                    if log_images:
                        # Re-creating the diff map logic from original script
                        diff = torch.abs(eval_target - eval_pred)
                        
                        metrics_to_log["ref_image"] = wandb.Image(
                            eval_target[0].detach().cpu().numpy().clip(0, 1), 
                            caption="Reference"
                        )
                        
                        metrics_to_log["reconstruction"] = wandb.Image(
                            eval_pred[0].detach().cpu().numpy().clip(0, 1), 
                            caption=f"Output (Epoch {iterator})"
                        )
                        
                        metrics_to_log["error_map"] = wandb.Image(
                            diff.detach().cpu().numpy().clip(0, 1), 
                            caption="Absolute Error"
                        )

                # ---------------------------------------------------------
                # 6. Logging and Checkpointing (Logic Preserved)
                # ---------------------------------------------------------
                wandb.log(metrics_to_log)  # type: ignore

                # Extract PSNR for tracking if it exists (most tasks have it)
                current_psnr = metrics.get('psnr', 0.0)
                psnrs.append(current_psnr)
                data_misfits.append(data_misfit)

                # Logic to determine best image based on data_misfit
                if len(data_misfits) == 1 or (len(data_misfits) > 1 and data_misfit < np.max(data_misfits[:-1])):
                    best_img = output_numpy.astype(float)
                    best_epoch = iterator
                
                # Save History to CSV
                df_new = pd.DataFrame([metrics_to_log])
                log_path = os.path.join(save_dir, rel_dir, "history.csv")
                if os.path.exists(log_path):
                    df_new.to_csv(log_path, mode='a', header=False, index=False)
                else:
                    df_new.to_csv(log_path, mode='w', header=True, index=False)
                
                # Early Stopping Logic
                if early_stop_indicator.get_flag() == False:
                    early_stop_indicator.update(loss.item() / loss_multiplier, output_numpy)
                else:
                    min_index = es_window_size - es_patience - 1
                    es_image = early_stop_indicator.get_images()[min_index]

                    np.save(os.path.join(save_dir, rel_dir, f"reconstruction_es_{str(iterator-es_window_size+min_index)}.npy"), es_image)
                    np.save(os.path.join(save_dir, rel_dir, f"reconstruction_best_{str(best_epoch)}.npy"), best_img)
                    break

        # ---------------------------------------------------------
        # 7. Final Saves (Post-Loop)
        # ---------------------------------------------------------
        # Save the raw data
        np.save(os.path.join(save_dir, rel_dir, "gt.npy"), ref_numpy)
        np.save(os.path.join(save_dir, rel_dir, f"reconstruction_best_{str(best_epoch)}.npy"), best_img)
        np.save(os.path.join(save_dir, rel_dir, f"reconstruction_last.npy"), output_numpy)

        if "fwi" in task:
            # FWI specific normalization for saving PNGs
            save_channels_png(fwi_norm(ref_numpy), os.path.join(save_dir, rel_dir, "gt"))
            save_channels_png(fwi_norm(best_img), os.path.join(save_dir, rel_dir, f"reconstruction_best_{str(best_epoch)}"))
            save_channels_png(fwi_norm(output_numpy), os.path.join(save_dir, rel_dir, "reconstruction_last"))
        else:
            save_channels_png(ref_numpy, os.path.join(save_dir, rel_dir, "gt"))
            save_channels_png(best_img, os.path.join(save_dir, rel_dir, f"reconstruction_best_{str(best_epoch)}"))
            save_channels_png(output_numpy, os.path.join(save_dir, rel_dir, "reconstruction_last"))

        # Save the config
        config_filename = os.path.join(save_dir, rel_dir, "config.yaml")
        with open(config_filename, "w") as file:
            yaml.safe_dump(config_all, file, default_flow_style=False)
