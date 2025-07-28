# stdlib
import glob
import inspect
import math
import os
import random
import time
from typing import List, Optional, Union

# third party
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
import tqdm
import wandb
import yaml  # type: ignore
from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
from huggingface_hub import login
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from torchvision import transforms
import pickle
# first party
from fmplug.losses.losses import PerceptualLossV3
import lpips
from fmplug.utils.measurements import get_noise, get_operator
from fmplug.utils.tv_norm import tv_lp_loss
from fmplug.utils.image_utils import Blurkernel, generate_tilt_map, mask_generator
from fmplug.utils.var_es import VarianceEarlyStopping, MeanEarlyStopping
import pandas as pd
import cv2

with open('poly13_model_var.pkl', 'rb') as f:
    reg = pickle.load(f)

# Setup device as cuda
device = torch.device("cuda")
scaler = torch.cuda.amp.GradScaler()

# Global variables for wandb
API_KEY = "bb47140c09574b488dcf0bc3d92f09d93b6241ce"
PROJECT_NAME = "FMPlug-Ada-DGP-ES"

# Enable wandb
print("Initialize Project ...")
wandb.login(key=API_KEY)  # type: ignore

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


def save_png_cv2(array, path):
    # If array is CHW, convert to HWC
    if array.ndim == 3 and array.shape[0] in [1, 3]:
        array = np.transpose(array, (1, 2, 0))

    # Clip and convert to uint8 if needed
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 1)
        array = (array * 255).round().astype(np.uint8)

    # If 3-channel RGB, convert to BGR for OpenCV
    if array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, array)

def visualize_image(ref: np.array, y_n: np.array, output: np.array, save_file_name: str) -> None:
    # Create a figure with 1 row, 3 columns
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 10))

    # Ground truth
    ax1.imshow(ref)
    ax1.set_title("Ground Truth Image")
    ax1.axis("off")

    # Display the corrupted image
    ax2.imshow(y_n)
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

def normalize_latent(z_x: torch.Tensor, t_x: float):
    """
    Normalize z_x at time t_x using interpolated mean and variance per channel.
    z_x: (B, C, H, W)
    t_x: scalar (float or 0-dim tensor)
    """
    z_var = reg(t_x.detach().cpu().numpy())
    z_var = torch.tensor(z_var, dtype=z_x.dtype, device=z_x.device)
    z_x = torch.sqrt(z_var / torch.var(z_x, unbiased=False)) * z_x
    return z_x

def integrate(
    f,
    z,
    t,
    NFE,
    prompt_embedding,
    pooled_embedding,
    device,
    guidance_scale: float = 7.0,
    method: str = "heun2"
):
    do_classifier_free_guidance = guidance_scale > 1.0
    # print("t: ", t)
    # print("NFE: ", NFE)
    # temp_t = 1000 * torch.sigmoid(t)
    temp_t = 1000 * torch.sigmoid(12.0 * t - 6.0)
    # print("temp_t: ", temp_t)
    delta_t = temp_t / NFE
    temp_t_next = temp_t - delta_t 
    
    sigma = temp_t / 1000
    sigma_next = temp_t_next / 1000
    zt = z
    zt = normalize_latent(zt, temp_t)

    for i in range(NFE):
        
        latent_model_input = zt
        time_step = temp_t.expand(latent_model_input.shape[0])
        time_step_next = temp_t_next.expand(latent_model_input.shape[0])
        if method == 'euler':
            noise_pred = f(
                x=latent_model_input,
                t=time_step,
                prompt_embedding=prompt_embedding,
                pooled_embedding=pooled_embedding,
                device=device,
            )

        elif method == 'heun2':
            dt = sigma_next - sigma
            k1 = f(
                x=latent_model_input,
                t=time_step,
                prompt_embedding=prompt_embedding,
                pooled_embedding=pooled_embedding,
                device=device,
            )

            # Predict next latent using Euler step
            x1_pred = latent_model_input + dt * k1

            # k2
            k2 = f(
                x=x1_pred,
                t=time_step_next,
                prompt_embedding=prompt_embedding,
                pooled_embedding=pooled_embedding,
                device=device,
            )

            # Heun2 step (average slope)
            noise_pred = 0.5 * dt * (k1 + k2)

        # # Rk4
        # half_dt = 0.5 * dt
        # k1 = f(
        #     x=latent_model_input,
        #     t=timestep,
        #     prompt_embedding=prompt_embedding,
        #     pooled_embedding=pooled_embedding,
        #     device=device,
        # )

        # k2 = f(
        #     x=(latent_model_input + half_dt * k1),
        #     t=(timestep + half_dt),
        #     prompt_embedding=prompt_embedding,
        #     pooled_embedding=pooled_embedding,
        #     device=device,
        # )

        # k3 = f(
        #     x=(latent_model_input + half_dt * k2),
        #     t=(timestep + half_dt),
        #     prompt_embedding=prompt_embedding,
        #     pooled_embedding=pooled_embedding,
        #     device=device,
        # )

        # k4 = f(
        #     x=(latent_model_input + dt * k3),
        #     t=prev_timestep,
        #     prompt_embedding=prompt_embedding,
        #     pooled_embedding=pooled_embedding,
        #     device=device,
        # )

        # noise_pred = (k1 + 2 * (k2 + k3) + k4) * dt * (1 / 6)
        # noise_pred = noise_pred.to(noise_pred.dtype)

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        else:
            noise_pred, noise_pred_text = noise_pred.chunk(2)

        # Update step for euler
        # prev_sample = sample + dt * noise_pred

        # Update step for huen2
        zt = zt + noise_pred
        
        temp_t = temp_t - delta_t 
        temp_t_next = temp_t - delta_t 
    
        sigma = temp_t / 1000
        sigma_next = temp_t_next / 1000
        # print("temp_t: ", temp_t)

    return zt


def solve(config_name: str) -> None:

    # Current memory usage by tensors (in MB)
    print("Init Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
    print("Init Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**2, 2), "MB")

    # Load in the configuration
    base_config_path = "/home/jusun/wan01530/Project/FMPlug/fmplug/configs"
    with open(os.path.join(base_config_path, config_name + ".yaml"), "r") as file:
        config_all = yaml.safe_load(file)

    # Now safely access each section
    fmplug_config = config_all["fmplug"]
    method = fmplug_config["method"]
    data_folder = fmplug_config["data_folder"]
    save_folder = fmplug_config["save_folder"]
    image_size = fmplug_config["img_size"]
    alpha = fmplug_config["alpha"]
    NFE = fmplug_config["NFE"]
    guidance_scale = fmplug_config["guidance_scale"]
    lr = fmplug_config["lr"]
    lr_dec = fmplug_config["lr_dec"]
    lr_t_ada = fmplug_config["lr_t_ada"]
    lr_alpha_ada = fmplug_config["lr_alpha_ada"]
    decay_factor =  fmplug_config["decay_factor"]
    epochs = fmplug_config["epochs"]
    loss_multiplier = fmplug_config["loss_multiplier"]
    loss_fn = fmplug_config["loss_fn"]
    vae_weight = fmplug_config["vae_weight"]
    lpips_weight = fmplug_config["lpips_weight"]
    TV_reg_weight = fmplug_config["TV_reg_weight"]
    t_end = fmplug_config["t_end"]
    data_type = eval(fmplug_config["data_type"])
    optimizer_select = fmplug_config["optimizer_select"]
    finetune_decoder_blocks_interval = fmplug_config.get("finetune_decoder_blocks_interval", 0) # Default to 0 (no incremental finetuning)
    
    es_window_size = fmplug_config["es_window_size"]
    es_patience = fmplug_config["es_patience"]
    es_min_epochs  = fmplug_config["es_min_epochs"]
    es_delta  = fmplug_config["es_delta"]
    
    
    measure_config = config_all['measurement']
    task = measure_config["operator"]["name"]

    gt_pattern = os.path.join(data_folder, task, "*", "*", "gt.png")
    gt_paths = sorted(glob.glob(gt_pattern))
    
    wandb_instance = wandb.init(  # type: ignore
            # set the wandb project where this run will be logged
            project=PROJECT_NAME,
            tags=["Experimental", task],
            config={
                "method": method,
                "optimizer": optimizer_select,
                "lr": lr,
                "lr_dec": lr_dec,
                "lr_t_ada": lr_t_ada,
                "decay_factor": decay_factor,
                "epochs": epochs,
                "loss_multiplier": loss_multiplier,
                "image_size": image_size,
                "NFE": NFE,
                "guidance_scale": guidance_scale,
                "loss_fn": loss_fn,
                "vae_weight": vae_weight,
                "lpips_weight": lpips_weight,
                "TV_reg_weight": TV_reg_weight,
                "t_end": t_end,
                "data_type": data_type,
            },
        )
    
    save_dir = os.path.join(save_folder, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    for gt_path in gt_paths:
        base_dir = os.path.dirname(gt_path)
        prompt_path = os.path.join(base_dir, "prompt.txt")

        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()

            # Get relative path starting from `task`
            rel_path = os.path.relpath(gt_path, start=data_folder)

            rel_dir = os.path.dirname(rel_path)  # gives: task/some_folder/another_folder

        else:
            print(f"Warning: prompt.txt not found for {gt_path}")
        gt_img = Image.open(gt_path).convert("RGB")

        tf = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )
        gt_img = tf(gt_img)

        ref_numpy = np.array(gt_img).transpose(1, 2, 0)
        x = gt_img * 2.0 - 1.0  # type: ignore

        ref_img = torch.Tensor(x).to(data_type).to(device).unsqueeze(0)
        ref_img.requires_grad = False

        # Forward measurement model (A(x) + n)
        operator = get_operator(device=device, **measure_config['operator'])
        noiser = get_noise(**measure_config['noise'])
        
        with torch.amp.autocast("cuda", dtype=data_type):
            if measure_config['operator']['name'] == 'inpainting':
                mask_gen = mask_generator(**measure_config['mask_opt'])
                mask = mask_gen(ref_img)[:, 0, :, :].unsqueeze(0)
                y = operator.forward(ref_img, mask=mask)
                y_n = noiser(y)
            elif measure_config['operator']['name'] == 'motion_blur' or measure_config['operator']['name'] == 'gaussian_blur':
                y = operator.forward(ref_img)
                y_n = noiser(y)
                kernel = operator.get_kernel()
            elif measure_config['operator']['name'] == 'turbulence':
                kernel_size = measure_config['kernel_opt']["kernel_size"]
                intensity = measure_config['kernel_opt']["intensity"]
                conv = Blurkernel('gaussian', kernel_size=kernel_size, device=device, std=intensity)
                kernel = conv.get_kernel().type(torch.float32)
                kernel = kernel.to(device).view(1, 1, kernel_size, kernel_size)
                tilt = generate_tilt_map(img_h=image_size, img_w=image_size, kernel_size=7, device=device)
                tilt = torch.clip(tilt, -2.5, 2.5)
                y = operator.forward(ref_img,kernel, tilt)
                y_n = noiser(y)
            else:
                y = operator.forward(ref_img)
                y_n = noiser(y)
        
        os.makedirs(os.path.join(save_dir, rel_dir), exist_ok=True)

        print("Load in SD3 image to image model ...")
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=data_type,
        )
        pipe = pipe.to(device)
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
                prompt=prompt,
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
        pooled_embedding = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )
            
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
        
        if optimizer_select == "adam":
            early_stop_indicator = VarianceEarlyStopping(es_window_size, es_patience, es_min_epochs, es_delta)
        elif optimizer_select == "lbfgs":
            early_stop_indicator = MeanEarlyStopping(es_window_size, es_patience, es_min_epochs, es_delta)

        # initialize latent
        
        y_n = y_n.detach().to(device).to(data_type)
        y_n_numpy = y_n.squeeze(0).detach().cpu().to(torch.float32).permute(1, 2, 0).numpy()
        y_n_numpy = (y_n_numpy + 1.0) / 2.0
        y_n.requires_grad = False

        if "super_resolution" in task:
            img = y_n.reshape([1, -1, image_size//int(measure_config['operator']['scale_factor']), image_size//int(measure_config['operator']['scale_factor'])])
            img = transforms.functional.resize(img, [image_size, image_size])
        else:
            img = y_n.reshape([1, -1, image_size, image_size])
        img = img.to(y_n.dtype)
        img = img.to(device)
        with torch.no_grad():
            # if "super_resolution" in task:
            #     blur = transforms.GaussianBlur(kernel_size=7, sigma=1.0)
            #     z = encode(blur(img))
            # else:
            #     z = encode(img)
            latent_y = encode(img)
            latent_y = latent_y.detach()
            # latent_y = (latent_y / torch.norm(latent_y, p=2) * math.sqrt(latent_y.numel())).detach()
            latent_y = latent_y.requires_grad_(False)
        
        del img

       
        
        z = torch.randn_like(latent_y)
        z = z / torch.norm(z, p=2) * math.sqrt(z.numel())
        z = torch.nn.parameter.Parameter(z, True).to(device)
        z = z.requires_grad_(True)
        # t_ada = torch.tensor(12.0 * (1.0 - alpha) - 6.0).to(device)
        t_ada = torch.tensor(1.0 - alpha).to(device)
        # t_ada = torch.tensor(-1.0).to(device)
        t_ada = t_ada.requires_grad_(True)
        
        alpha_ada = torch.tensor(alpha).to(device)
        alpha_ada = alpha_ada.requires_grad_(True)

        # amplitude = torch.tensor(torch.pi).to(dtype=data_type, device=device)
        
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

        def checkpointed_integrate(z):
            return integrate(
                f,
                z,
                t_ada,
                NFE,
                prompt_embedding,
                pooled_embedding,
                device,
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

        # Setup perceptual loss
        # percep_loss_fn = PerceptualLoss(layers=["relu1_2"]).to(device)
        percep_loss_fn = PerceptualLossV3(layers=["relu1_2"]).to(device)
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
        # style_loss_fn = StyleLoss(layers=["relu1_2"]).to(device)
        # fft_loss_fn = FFTLoss()
        
        L1_tv = tv_lp_loss(pow = 1)
        L2_tv = tv_lp_loss(pow = 2)
        
        if optimizer_select == "adam":
            params_group1 = {'params': z, 'lr': lr[0]}
            params_group2 = {'params': t_ada, 'lr': lr_t_ada}
            params_group3 = {'params': alpha_ada, 'lr': lr_alpha_ada}
            
            # Get decoder blocks for incremental fine-tuning
            decoder_blocks = list(vae.decoder.up_blocks) # Assuming decoder blocks are in vae.decoder.up_blocks
            num_decoder_blocks = len(decoder_blocks)
            
            
            # unfreeze_schedule = [
            #                     vae.decoder.conv_in,
            #                     vae.decoder.mid_block,
            #                     vae.decoder.up_blocks[0],
            #                     vae.decoder.up_blocks[1],
            #                     vae.decoder.up_blocks[2],
            #                     vae.decoder.up_blocks[3],
            #                     vae.decoder.conv_norm_out,
            #                     vae.decoder.conv_act,
            #                     vae.decoder.conv_out,
            #                 ]

            
            # Initialize optimizer with z and t_ada, no decoder params initially
            optimizer = torch.optim.AdamW([params_group1, params_group2, params_group3])
            

        elif optimizer_select == "lbfgs":
            optimizer_z = torch.optim.LBFGS([z], lr=lr[0], max_iter=20, history_size=20, line_search_fn="strong_wolfe")
            optimizer_t = torch.optim.LBFGS([t_ada], lr=lr_t_ada, max_iter=20, history_size=20, line_search_fn="strong_wolfe")
        

        psnrs = []
        losses = []
        best_images = []
        z_grad_norms = []
        z_deltas = []
        z_rel_updates = []
        z_prev = z.clone().detach()
        rel_update, delta, grad_norm = 0.0, 0.0, 0.0
        
        # Current memory usage by tensors (in MB)
        print("Init Opt Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
        print("Init Opt Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**2, 2), "MB")
        decoded_output = None
        d = math.sqrt(z.numel())
        def closure():
            nonlocal decoded_output
            optimizer.zero_grad()
            # torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast("cuda", dtype=data_type):
                temp_alpha = torch.sigmoid((alpha_ada-0.5)*6)
                temp_z = (1-temp_alpha) * z + temp_alpha * latent_y
                # temp_z = torch.sqrt(1-temp_alpha) * z + torch.sqrt(temp_alpha) * latent_y
                # with torch.no_grad():
                #     temp_z.data = temp_z / torch.norm(temp_z, p=2) * d
                x_t = checkpoint(checkpointed_integrate, temp_z)
                x_t = (x_t / vae.config.scaling_factor) + vae.config.shift_factor
                decoded_output = torch.sin(vae.decode(x_t).sample)

                if measure_config['operator']['name'] == 'inpainting':
                    operator_decoded_output = operator.forward(decoded_output, mask=mask)
                else:
                    operator_decoded_output = operator.forward(decoded_output)

                loss = criterion(operator_decoded_output, y_n)
                encoded = vae.encode(decoded_output).latent_dist.sample()
                loss += vae_weight * criterion(x_t, encoded)
                loss += lpips_weight * percep_loss_fn((operator_decoded_output + 1.0) / 2.0, (y_n + 1.0) / 2.0)
                # loss += lpips_weight * lpips_loss_fn((operator_decoded_output + 1.0) / 2.0, (y_n + 1.0) / 2.0).mean()
                loss += TV_reg_weight * L1_tv(decoded_output) / L2_tv(decoded_output) / decoded_output.numel() / 2.0
                loss *= loss_multiplier
            
            loss = loss.float()
            # scaler.scale(loss).backward()
            loss.backward()
            # Report peak memory used in MB
            # peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            # print(f"Peak GPU memory used: {peak_memory:.2f} MB")
            # print("gradient z: ", z.grad.min(), z.grad.max())
            
            # During training loop:
            

            # Backward + optimizer.step()

            return loss


        def closure_z():
            nonlocal decoded_output
            optimizer_z.zero_grad()
            # torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast("cuda", dtype=data_type):
                temp_z = alpha_ada * z + (1 - torch.sigmoid((alpha_ada-0.5)*6)) * latent_y
                x_t = checkpoint(checkpointed_integrate, temp_z)
                x_t = (x_t / vae.config.scaling_factor) + vae.config.shift_factor
                decoded_output = torch.sin(vae.decode(x_t).sample)

                if measure_config['operator']['name'] == 'inpainting':
                    operator_decoded_output = operator.forward(decoded_output, mask=mask)
                else:
                    operator_decoded_output = operator.forward(decoded_output)

                loss = criterion(operator_decoded_output, y_n)
                encoded = vae.encode(decoded_output).latent_dist.sample()
                loss += vae_weight * criterion(x_t, encoded)
                loss += lpips_weight * percep_loss_fn((operator_decoded_output + 1.0) / 2.0, (y_n + 1.0) / 2.0)
                loss += TV_reg_weight * L1_tv(decoded_output) / L2_tv(decoded_output) / decoded_output.numel() / 2.0
                loss *= loss_multiplier
            
            loss = loss.float()
            scaler.scale(loss).backward()
            # Report peak memory used in MB
            # peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            # print(f"Peak GPU memory used: {peak_memory:.2f} MB")
            # print("gradient z: ", z.grad.min(), z.grad.max())

            return loss


        def closure_t():
            nonlocal decoded_output
            optimizer_t.zero_grad()
            # torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast("cuda", dtype=data_type):
                temp_z = alpha_ada * z + (1 - torch.sigmoid((alpha_ada-0.5)*6)) * latent_y
                x_t = checkpoint(checkpointed_integrate, temp_z)
                x_t = (x_t / vae.config.scaling_factor) + vae.config.shift_factor
                decoded_output = torch.sin(vae.decode(x_t).sample)

                if measure_config['operator']['name'] == 'inpainting':
                    operator_decoded_output = operator.forward(decoded_output, mask=mask)
                else:
                    operator_decoded_output = operator.forward(decoded_output)

                loss = criterion(operator_decoded_output, y_n)
                encoded = vae.encode(decoded_output).latent_dist.sample()
                loss += vae_weight * criterion(x_t, encoded)
                loss += lpips_weight * percep_loss_fn((operator_decoded_output + 1.0) / 2.0, (y_n + 1.0) / 2.0)
                loss += TV_reg_weight * L1_tv(decoded_output) / L2_tv(decoded_output) / decoded_output.numel() / 2.0
                loss *= loss_multiplier
            
            loss = loss.float()
            scaler.scale(loss).backward()
            # Report peak memory used in MB
            # peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            # print(f"Peak GPU memory used: {peak_memory:.2f} MB")
            # print("gradient z: ", z.grad.min(), z.grad.max())

            return loss

        # total_start_time = time.time()

        for iterator in tqdm.tqdm(range(epochs)):
            iter_start_time = time.time()
            # Incremental fine-tuning of VAE decoder blocks
            # if finetune_decoder_blocks_interval > 0 and iterator % finetune_decoder_blocks_interval == 0 and iterator < finetune_decoder_blocks_interval * num_decoder_blocks:
            if finetune_decoder_blocks_interval > 0 and iterator > 0 and iterator % finetune_decoder_blocks_interval == 0 and iterator < finetune_decoder_blocks_interval * len(lr_dec):
                block_to_unfreeze_idx = iterator // finetune_decoder_blocks_interval - 1
                if block_to_unfreeze_idx < num_decoder_blocks:
                    if lr_dec[block_to_unfreeze_idx] != 0.0:
                        for param in decoder_blocks[block_to_unfreeze_idx].resnets.parameters():
                            param.requires_grad_(True)
                        print(f"Unfreezing decoder block {block_to_unfreeze_idx} to optimizer with learning rate {lr_dec[block_to_unfreeze_idx]}")
                        optimizer.add_param_group({'params': decoder_blocks[block_to_unfreeze_idx].resnets.parameters(), 'lr': lr_dec[block_to_unfreeze_idx]})
                        
                    # optimizer.param_groups[0]['lr'] = lr[block_to_unfreeze_idx]
                    optimizer.param_groups[0]['lr'] = lr[block_to_unfreeze_idx+1]
                    # for param_group_idx in range(1, block_to_unfreeze_idx + 1):
                    #     # Update the learning rate for previously added decoder blocks
                    #     if optimizer.param_groups[-param_group_idx]['lr'] != 0.0:
                    #         optimizer.param_groups[-param_group_idx]['lr'] = lr_dec[block_to_unfreeze_idx]
                # if block_to_unfreeze_idx == 0:
                #     for param in decoder_blocks[-1].parameters():
                #         param.requires_grad_(True)
                #     optimizer.add_param_group({'params': decoder_blocks[-1].parameters(), 'lr': lr_dec[block_to_unfreeze_idx]})
                # else:
                #     optimizer.param_groups[-1]['lr'] = lr_dec[block_to_unfreeze_idx]
            
            
            if optimizer_select == "adam":
                loss = optimizer.step(closure)
                new_lr = lr_t_ada * (decay_factor ** iterator)
                optimizer.param_groups[1]['lr'] = new_lr
            elif optimizer_select == "lbfgs":
                _ = optimizer_z.step(closure_z)
                loss = optimizer_t.step(closure_t)
                
            grad_norm = z.grad.norm().item()
            delta = (z - z_prev).norm().item()
            rel_update = delta / (z_prev.norm() + 1e-8)

            z_grad_norms.append(grad_norm)
            z_deltas.append(delta)
            z_rel_updates.append(rel_update.item())
            z_prev = z.clone().detach()
            # scheduler.step()
            with torch.no_grad():
                z.data = z / torch.norm(z, p=2) * math.sqrt(z.numel())
            losses.append(loss.item())
            
            # print("Opt Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
            # print("Opt Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**2, 2), "MB")
            torch.cuda.empty_cache()
            iter_time = time.time() - iter_start_time
            # print(f"Iter Time = {iter_time:.4f} sec")
            
            # Evaluate
            with torch.no_grad():
                output = decoded_output.detach().float()
                lpips_score = lpips_loss_fn(output, ref_img).mean()
                output_numpy = np.clip((output.cpu().squeeze().numpy() + 1) / 2, 0, 1)
                output_numpy = np.transpose(
                    output_numpy, (1, 2, 0)
                )  # Keep out for now lets evaluate

                # calculate psnr
                tmp_psnr = peak_signal_noise_ratio(
                    ref_numpy, output_numpy
                )
                # print(tmp_psnr)

                # calculate mse
                mse_score = ((ref_numpy - output_numpy) ** 2).mean()

                metrics_to_log = {
                    "epoch": iterator,
                    "t_ada": (1000 * torch.sigmoid(12.0 * t_ada - 6.0)).item(),
                    "alpha_ada": (torch.sigmoid((alpha_ada-0.5)*6)).item(),
                    "loss": loss.item()/loss_multiplier,
                    "psnr": tmp_psnr,
                    "lpips": lpips_score.item(),
                    "mse_loss": mse_score,
                    # "z_grad_norms": grad_norm,
                    # "z_deltas": delta,
                    "z_rel_updates": rel_update.item(),
                    "iter_time": iter_time,
                }
                wandb.log(metrics_to_log)  # type: ignore

                psnrs.append(tmp_psnr)

                if len(psnrs) == 1 or (len(psnrs) > 1 and tmp_psnr > np.max(psnrs[:-1])):
                    best_img = output_numpy.astype(float)
                    best_images.append(best_img)
                    best_epoch = iterator
                
                df_new = pd.DataFrame([metrics_to_log])
                # Append or create
                log_path = os.path.join(save_dir, rel_dir, "history.csv")
                if os.path.exists(log_path):
                    df_new.to_csv(log_path, mode='a', header=False, index=False)
                else:
                    df_new.to_csv(log_path, mode='w', header=True, index=False)
                
                if early_stop_indicator.get_flag() == False:
                    if optimizer_select == "adam":
                        early_stop_indicator.update(loss.item() / loss_multiplier, output_numpy)
                    elif optimizer_select == "lbfgs":
                        early_stop_indicator.update(rel_update.item(), output_numpy)
                else:
                    # min_index = min(range(len(early_stop_indicator.get_losses())), key=lambda i: losses[i])
                    min_index = es_window_size - es_patience - 1

                    es_image = early_stop_indicator.get_images()[min_index]

                    visualize_image(ref_numpy, y_n_numpy, best_img, os.path.join(save_dir, rel_dir, f"img_diff_es_{str(iterator-es_window_size+min_index)}.png"))
                    np.save(os.path.join(save_dir, rel_dir, f"reconstruction_es_{str(iterator-es_window_size+min_index)}.npy"), es_image)
                    save_png_cv2(es_image, os.path.join(save_dir, rel_dir, f"reconstruction_es_{str(iterator-es_window_size+min_index)}.png"))
                    break

        # total_time = time.time() - total_start_time
        # print(f"\nTotal optimization time: {total_time:.2f} sec")
        visualize_image(ref_numpy, y_n_numpy, best_img, os.path.join(save_dir, rel_dir, f"img_diff_best.png"))
        visualize_image(ref_numpy, y_n_numpy, output_numpy, os.path.join(save_dir, rel_dir, f"img_diff_last.png"))
        
        # Save the raw data as well
        np.save(os.path.join(save_dir, rel_dir, "gt.npy"), ref_numpy)
        np.save(os.path.join(save_dir, rel_dir, "measurement.npy"), y_n_numpy)
        np.save(os.path.join(save_dir, rel_dir, f"reconstruction_best_{str(best_epoch)}.npy"), best_img)
        np.save(os.path.join(save_dir, rel_dir, f"reconstruction_last.npy"), output_numpy)
        
        save_png_cv2(ref_numpy, os.path.join(save_dir, rel_dir, "gt.png"))
        save_png_cv2(y_n_numpy, os.path.join(save_dir, rel_dir, "measurement.png"))
        save_png_cv2(best_img, os.path.join(save_dir, rel_dir, f"reconstruction_best_{str(best_epoch)}.png"))
        save_png_cv2(output_numpy, os.path.join(save_dir, rel_dir, f"reconstruction_last.png"))
        
        
        with open(os.path.join(save_dir, rel_dir, "prompt.txt"), "w") as file:
            file.write(prompt)

        # Save the file
        config_filename = os.path.join(save_dir, rel_dir, "config.yaml")
        with open(config_filename, "w") as file:
            yaml.safe_dump(config_all, file, default_flow_style=False)
