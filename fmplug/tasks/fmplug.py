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
import yaml  # type: ignore
from diffusers import StableDiffusion3Img2ImgPipeline
from huggingface_hub import login
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from torchvision import transforms
import pandas as pd

# first party
import lpips
from fmplug.utils.img_save_helper import visualize_image, save_png_cv2
from fmplug.utils.measurements import get_noise, get_operator
from fmplug.utils.image_utils import mask_generator
from fmplug.utils.var_es import VarianceEarlyStopping
from fmplug.utils.regularization import gauss_sphere_reg


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

# Setup device as cuda
device = torch.device("cuda")

set_seed(123)  # Set a fixed seed for reproducibility



def integrate(
    f,
    z,
    t,
    NFE,
    prompt_embedding,
    pooled_prompt_embedding,
    device,
    guidance_scale: float = 7.0,
    method: str = "heun2"
):
    do_classifier_free_guidance = guidance_scale > 1.0
    temp_t = 1000 * t
    delta_t = temp_t / NFE
    temp_t_next = temp_t - delta_t 
    
    sigma = temp_t / 1000
    sigma_next = temp_t_next / 1000
    zt = z
        
    latent_model_input = torch.cat([zt] * 2) if do_classifier_free_guidance else zt

    for i in range(NFE):
        
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
            # print("Step: ", i, "temp_k1: ", temp_k1.requires_grad)
            # Predict next latent using Euler step
            x1_pred = latent_model_input + dt * k1
            # print("Step: ", i, "x1_pred: ", x1_pred.requires_grad)

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
            # print("Step: ", i, "temp_k2: ", temp_k2.requires_grad)
            
            # Heun2 step (average slope)
            noise_pred = 0.5 * dt * (k1 + k2)
            # print("Step: ", i, "noise_pred: ", noise_pred.requires_grad)

            

        # Update step for euler
        # prev_sample = sample + dt * noise_pred

        # Update step for huen2
        zt = zt + noise_pred
        # print("Step: ", i, "zt: ", zt.requires_grad)
        
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
    base_config_path = "./fmplug/configs"
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
    lr_z = fmplug_config["lr_z"]
    lr_dec = fmplug_config["lr_dec"]
    lr_alpha_ada = fmplug_config["lr_alpha_ada"]
    decay_factor =  fmplug_config["decay_factor"]
    epochs = fmplug_config["epochs"]
    loss_multiplier = fmplug_config["loss_multiplier"]
    loss_fn = fmplug_config["loss_fn"]
    is_gauss_reg = fmplug_config["is_gauss_reg"]
    t_end = fmplug_config["t_end"]
    data_type = eval(fmplug_config["data_type"])
    optimizer_select = fmplug_config["optimizer_select"]
    
    es_window_size = fmplug_config["es_window_size"]
    es_patience = fmplug_config["es_patience"]
    es_min_epochs  = fmplug_config["es_min_epochs"]
    es_delta  = fmplug_config["es_delta"]
    
    
    measure_config = config_all['measurement']
    task = measure_config["operator"]["name"]

    gt_pattern = os.path.join(data_folder, task, "*", "*", "gt.png")
    gt_paths = sorted(glob.glob(gt_pattern))
    
    
    save_dir = os.path.join(save_folder, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    
    for gt_path in gt_paths:
        base_dir = os.path.dirname(gt_path)
        prompt_path = os.path.join(base_dir, "prompt.txt")

        # Always compute a safe relative output directory, even if prompt.txt is missing.
        rel_path = os.path.relpath(gt_path, start=data_folder)
        rel_dir = os.path.dirname(rel_path) or "."

        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            print(f"Warning: prompt.txt not found for {gt_path}; using an empty prompt.")
            prompt = ""

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
            else:
                y = operator.forward(ref_img)
                y_n = noiser(y)
        
        os.makedirs(os.path.join(save_dir, rel_dir), exist_ok=True)

        print("Load in SD3 image to image model ...")
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            text_encoder_3=None,
            tokenizer_3=None,
            dtype=data_type,
        )
        pipe = pipe.to(device, dtype=data_type)
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
        vae.decoder.requires_grad_(False)
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
        
        early_stop_indicator = VarianceEarlyStopping(es_window_size, es_patience, es_min_epochs, es_delta)

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
            latent_y = encode(img)
            latent_y = latent_y.detach()
            latent_y = latent_y.requires_grad_(False)
        del img
       
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
        z = torch.randn_like(latent_y)
        z = torch.nn.parameter.Parameter(z, True).to(device)
        z = z.requires_grad_(True)

        
        alpha_ada = torch.tensor(alpha).to(device)
        alpha_ada = alpha_ada.requires_grad_(True)
        t_ada = (1 - torch.sigmoid((alpha_ada-0.5)*6))

        
        # Solve the ODE
        print("Solve inverse problem ...")

        def f(x, t, prompt_embedding, pooled_embedding, device):
            with torch.amp.autocast(device.type, dtype=data_type):
                return transformer(
                    hidden_states=x,
                    timestep=t,
                    encoder_hidden_states=prompt_embedding,
                    pooled_projections=pooled_embedding,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0].to(torch.float32)


        def checkpointed_integrate(z, t):
            return integrate(
                f,
                z,
                t,
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
        
        if optimizer_select == "adam":
            params_group1 = {'params': z, 'lr': lr_z[0]}
            params_group2 = {'params': alpha_ada, 'lr': lr_alpha_ada}
            optimizer = torch.optim.AdamW([params_group1, params_group2])

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


        for iterator in tqdm.tqdm(range(epochs)):
            with torch.no_grad():
                z = gauss_sphere_reg(z) if is_gauss_reg else z # TODO: randomness for stability
            def closure():
                nonlocal decoded_output
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", dtype=data_type):
                    temp_alpha = torch.sigmoid((alpha_ada-0.5)*6)
                    temp_z = (1-temp_alpha) * z + temp_alpha * latent_y
                    # x_t = checkpointed_integrate(temp_z, t_ada)
                    x_t = checkpoint(checkpointed_integrate, temp_z, t_ada)
                    decoded_output = decode(x_t)
                    if measure_config['operator']['name'] == 'inpainting':
                        operator_decoded_output = operator.forward(decoded_output, mask=mask)
                    else:
                        operator_decoded_output = operator.forward(decoded_output)

                    loss = criterion(operator_decoded_output, y_n)
                    loss *= loss_multiplier
                
                loss = loss.float()
                loss.backward()
                return loss
            
            
            loss = optimizer.step(closure)
            new_lr = lr_alpha_ada * (decay_factor ** iterator)
            optimizer.param_groups[1]['lr'] = new_lr
             

            t_ada = (1 - torch.sigmoid((alpha_ada-0.5)*6))
            
            grad_norm = z.grad.norm().item()
            delta = (z - z_prev).norm().item()
            rel_update = delta / (z_prev.norm() + 1e-8)

            z_grad_norms.append(grad_norm)
            z_deltas.append(delta)
            z_rel_updates.append(rel_update.item())
            z_prev = z.clone().detach()
            
            losses.append(loss.item())
            
            torch.cuda.empty_cache()
            
            # Evaluate
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=data_type):
                    z_reg = gauss_sphere_reg(z) if is_gauss_reg else z
                    temp_alpha = torch.sigmoid((alpha_ada-0.5)*6)
                    temp_z = (1-temp_alpha) * z_reg + temp_alpha * latent_y
                    x_t = checkpointed_integrate(temp_z, t_ada)
                    decoded_output = decode(x_t)
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

                psnrs.append(tmp_psnr)
                
                metrics_to_log = {
                    "epoch": iterator,
                    "t_ada": 1000 * (1 - torch.sigmoid((alpha_ada-0.5)*6)).item(),
                    "alpha_ada": (torch.sigmoid((alpha_ada-0.5)*6)).item(),
                    "loss": loss.item()/loss_multiplier,
                    "psnr": tmp_psnr,
                    "lpips": lpips_score.item(),
                    "mse_loss": mse_score,
                    # "z_grad_norms": grad_norm,
                    # "z_deltas": delta,
                    "z_rel_updates": rel_update.item(),
                }

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
                    early_stop_indicator.update(loss.item() / loss_multiplier, output_numpy)
                else:
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