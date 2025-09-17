# stdlib
import glob
import inspect
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
from fmplug.utils.var_es import VarianceEarlyStopping
from fmplug.dip_module.UNet import UNet
from fmplug.dip_module.Siren import Siren, get_mgrid
from fmplug.dip_module.skip import skip

import pandas as pd
import cv2

with open('poly13_model_var.pkl', 'rb') as f:
    reg = pickle.load(f)

# Setup device as cuda
device = torch.device("cuda")
scaler = torch.cuda.amp.GradScaler()

# Global variables for wandb
API_KEY = "bb47140c09574b488dcf0bc3d92f09d93b6241ce"
PROJECT_NAME = "DIP"

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
    scale_factor = fmplug_config["scale_factor"]
    alpha = fmplug_config["alpha"]
    NFE = fmplug_config["NFE"]
    guidance_scale = fmplug_config["guidance_scale"]
    lr = fmplug_config["lr"]
    lr_t_ada = fmplug_config["lr_t_ada"]
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
    
    es_window_size = fmplug_config["es_window_size"]
    es_patience = fmplug_config["es_patience"]
    es_min_epochs  = fmplug_config["es_min_epochs"]
    es_delta  = fmplug_config["es_delta"]
    
    dip_in_channel = fmplug_config["dip_in_channel"]
    dip_out_channel = fmplug_config["dip_out_channel"]
    dip_num_channels_down = fmplug_config["dip_num_channels_down"]
    dip_num_channels_up = fmplug_config["dip_num_channels_up"]
    dip_num_channels_skip = fmplug_config["dip_num_channels_skip"]
    dip_filter_size_down = fmplug_config["dip_filter_size_down"]
    dip_filter_size_up = fmplug_config["dip_filter_size_up"]
    dip_filter_skip_size = fmplug_config["dip_filter_skip_size"]
    dip_upsample_mode = fmplug_config["dip_upsample_mode"]
    dip_need_sigmoid = fmplug_config["dip_need_sigmoid"]
    dip_need_bias = fmplug_config["dip_need_bias"]
    dip_pad = fmplug_config["dip_pad"]
    dip_act_fun = fmplug_config["dip_act_fun"]
    dip_lr = fmplug_config["dip_lr"]
    
    siren_in_features = fmplug_config["siren_in_features"]
    siren_hidden_features = fmplug_config["siren_hidden_features"]
    siren_out_features = fmplug_config["siren_out_features"]
    siren_hidden_layers = fmplug_config["siren_hidden_layers"]
    siren_outermost_linear = fmplug_config["siren_outermost_linear"]
    
    target_norm = fmplug_config["target_norm"]
    
    measure_config = config_all['measurement']
    task = measure_config["operator"]["name"]

    gt_pattern = os.path.join(data_folder, task, "*", "*", "gt.png")
    gt_paths = sorted(glob.glob(gt_pattern))
    
    wandb_instance = wandb.init(  # type: ignore
            # set the wandb project where this run will be logged
            project=PROJECT_NAME,
            tags=["Experimental", task],
            config={
                "epochs": epochs,
                "loss_multiplier": loss_multiplier,
                "image_size": image_size,
                "scale_factor": scale_factor,
                "loss_fn": loss_fn,
                "lpips_weight": lpips_weight,
                "TV_reg_weight": TV_reg_weight,
                "t_end": t_end,
                "data_type": data_type
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
        
        kernel_size = measure_config['kernel_opt']["kernel_size"]
        intensity = measure_config['kernel_opt']["intensity"]
    
        tilt = generate_tilt_map(img_h=image_size, img_w=image_size, kernel_size=7, device=device)
        tilt = torch.clip(tilt, -2.5, 2.5)
        # blur kernel
        conv = Blurkernel('gaussian', kernel_size=kernel_size, device=device, std=intensity)
        kernel = conv.get_kernel().type(torch.float32)
        kernel = kernel.to(device).view(1, 1, kernel_size, kernel_size)
        with torch.amp.autocast("cuda", dtype=data_type):
            y = operator.forward(ref_img,kernel,tilt)
            y_n = noiser(y)
        y_n.requires_grad = False
        
        os.makedirs(os.path.join(save_dir, rel_dir), exist_ok=True)

        
        early_stop_indicator = VarianceEarlyStopping(es_window_size, es_patience, es_min_epochs, es_delta)

        # initialize latent
        
        y_n = y_n.detach().to(device).to(data_type)
        y_n_numpy = y_n.squeeze(0).detach().cpu().to(torch.float32).permute(1, 2, 0).numpy()
        y_n_numpy = (y_n_numpy + 1.0) / 2.0
        y_n.requires_grad = False
        
        img_z = torch.randn(1, dip_in_channel, image_size, image_size, device=device, dtype=data_type).to(device).requires_grad_(False)
        img_rep = skip(  
                        num_input_channels=dip_in_channel,
                        num_output_channels=dip_out_channel,
                        num_channels_down=dip_num_channels_down,
                        num_channels_up=dip_num_channels_up,
                        num_channels_skip=dip_num_channels_skip,
                        filter_size_down=dip_filter_size_down,
                        filter_size_up=dip_filter_size_up,
                        filter_skip_size=dip_filter_skip_size,
                        upsample_mode=dip_upsample_mode,
                        need_sigmoid=dip_need_sigmoid, 
                        need_bias=dip_need_bias, 
                        pad=dip_pad, 
                        act_fun=dip_act_fun).to(device)
        
        trainable_kernel = torch.randn((1, kernel_size * kernel_size), device=device, dtype=data_type)
        trainable_tilt = torch.randn((1, 2, image_size, image_size), device=device, dtype=data_type) * 0.01
        trainable_kernel.requires_grad = True
        trainable_tilt.requires_grad = True
        # img_z = get_mgrid([1, 3, image_size, image_size]).to(device)
        # img_rep = Siren(
        #                 in_features=siren_in_features, 
        #                 hidden_features=siren_hidden_features, 
        #                 out_features=siren_out_features, 
        #                 hidden_layers=siren_hidden_layers, 
        #                 outermost_linear=siren_outermost_linear).to(device)

        # amplitude = torch.tensor(torch.pi).to(dtype=data_type, device=device)
        
        # Solve the ODE
        print("Solve inverse problem ...")

        # Criterion for learning
        if loss_fn == "l1":
            criterion = torch.nn.L1Loss().to(device)

        elif loss_fn == "mse":
            criterion = torch.nn.MSELoss().to(device)  # type: ignore

        # Setup perceptual loss
        # percep_loss_fn = PerceptualLoss(layers=["relu1_2"]).to(device)
        percep_loss_fn = PerceptualLossV3(layers=["relu1_2"]).to(device)
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
        # style_loss_fn = StyleLoss(layers=["relu1_2"]).to(device)
        # fft_loss_fn = FFTLoss()
        
        L1_tv = tv_lp_loss(pow = 1)
        L2_tv = tv_lp_loss(pow = 2)
        
        if optimizer_select == "adam":
            params_group1 = {'params': img_rep.parameters(), 'lr': dip_lr}
            params_group2 = {'params': trainable_kernel, 'lr': 0.1}
            params_group3 = {'params': trainable_tilt, 'lr': 1E-7}
            dip_optimizer = torch.optim.AdamW([params_group1, params_group2, params_group3])

        

        psnrs = []
        losses = []
        best_images = []
        
        # Current memory usage by tensors (in MB)
        print("Init Opt Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
        print("Init Opt Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**2, 2), "MB")
        img_dip_rep = None
        
        
        def dip_closure():
            nonlocal img_dip_rep
            dip_optimizer.zero_grad()
            # torch.cuda.reset_peak_memory_stats()
            with torch.amp.autocast("cuda", dtype=data_type):
                img_dip_rep = torch.tanh(img_rep(img_z + torch.randn_like(img_z).detach() * 0.03))  # DIP output
                kernel_output = torch.nn.functional.softmax(trainable_kernel, dim=1)
                out_k = kernel_output.view(1, 1, kernel_size, kernel_size)
                operator_dip_output = operator.forward(img_dip_rep, out_k, trainable_tilt)
                loss = criterion(operator_dip_output, y_n)
                # loss += lpips_weight * percep_loss_fn((operator_dip_output + 1.0) / 2.0, (y_n + 1.0) / 2.0)
                # loss += TV_reg_weight * L1_tv(img_dip_rep) / L2_tv(img_dip_rep) / img_dip_rep.numel() / 2.0
                # loss *= loss_multiplier
            
            loss.backward()
            # scaler.scale(loss).backward()
            # Report peak memory used in MB
            # peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            # print(f"Peak GPU memory used: {peak_memory:.2f} MB")
            # print("gradient z: ", z.grad.min(), z.grad.max())

            return loss


        for iterator in tqdm.tqdm(range(epochs)):
            # print("Iter: ", iterator)
            if optimizer_select == "adam":
                loss = dip_optimizer.step(dip_closure)
            # scheduler.step()

            losses.append(loss.item())
            
            # print("Opt Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
            # print("Opt Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**2, 2), "MB")
            torch.cuda.empty_cache()
            
            # Evaluate
            with torch.no_grad():
                output = img_dip_rep.detach().float()
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
                    "loss": loss.item()/loss_multiplier,
                    "psnr": tmp_psnr,
                    "lpips": lpips_score.item(),
                    "mse_loss": mse_score,
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
                    early_stop_indicator.update(loss.item(), output_numpy)
                else:
                    # min_index = min(range(len(early_stop_indicator.get_losses())), key=lambda i: losses[i])
                    min_index = es_window_size - es_patience - 1

                    es_image = early_stop_indicator.get_images()[min_index]

                    visualize_image(ref_numpy, y_n_numpy, best_img, os.path.join(save_dir, rel_dir, f"img_diff_es_{str(iterator-es_window_size+min_index)}.png"))
                    np.save(os.path.join(save_dir, rel_dir, f"reconstruction_es_{str(iterator-es_window_size+min_index)}.npy"), es_image)
                    save_png_cv2(es_image, os.path.join(save_dir, rel_dir, f"reconstruction_es_{str(iterator-es_window_size+min_index)}.png"))
                    break


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
