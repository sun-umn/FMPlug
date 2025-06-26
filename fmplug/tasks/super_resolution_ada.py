# stdlib
import inspect
import os
import random
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
from fmplug.layers.activations import CoordFeatureSiren
from fmplug.losses.losses import PerceptualLossV3
import lpips
from fmplug.utils.measurements import get_noise, get_operator
from fmplug.utils.tv_norm import tv_lp_loss

with open('poly13_model_var.pkl', 'rb') as f:
    reg = pickle.load(f)

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


def soft_clamp(x, min_val=0.0, max_val=1000.0, slope=0.01):
    return min_val + (max_val - min_val) * torch.sigmoid(slope * (x - min_val))


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves
    timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples
            with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`,
            the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy
            of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of
            the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is
        the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. "
            "Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s "
                "`set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using"
                "the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)  # type: ignore
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s"
                "`set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether"
                "you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@torch.compile
def linear_interp(t0, t1, y0, y1, t):
    if t == t0:
        return y0

    if t == t1:
        return y1

    slope = (t - t0) / (t1 - t0)
    return y0 + slope * (y1 - y0)


def make_exponential_timesteps(t_start=925.0, t_end=0.5, num_steps=5, decay=5.0):
    s = torch.linspace(0, 1, num_steps + 1)
    decay_values = torch.exp(-decay * s)

    # Normalize to range [0, 1]
    decay_values = (decay_values - decay_values[-1]) / (
        decay_values[0] - decay_values[-1]
    )

    # Linearly map to [t_end, t_start]
    timesteps = t_end + (t_start - t_end) * decay_values
    return timesteps



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
    print("t: ", t)
    print("NFE: ", NFE)
    temp_t = 1000 * torch.sigmoid(t)
    print("temp_t: ", temp_t)
    delta_t = temp_t / NFE
    temp_t_next = temp_t - delta_t 
    
    sigma = temp_t / 1000
    sigma_next = temp_t_next / 1000
    zt = z
    zt = normalize_latent(zt, temp_t)

    for i in range(NFE):
        
        latent_model_input = torch.cat([zt] * 2) if do_classifier_free_guidance else zt
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
        print("temp_t: ", temp_t)

    return zt


def super_resolution_task(config_name: str) -> None:

    # Current memory usage by tensors (in MB)
    print("Init Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
    print("Init Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**2, 2), "MB")

    # Load in the configuration
    base_config_path = "/scratch.global/wan01530/FMPlug/fmplug/configs"
    with open(os.path.join(base_config_path, config_name + ".yaml"), "r") as file:
        config_all = yaml.safe_load(file)

    # Now safely access each section
    config = config_all["fmplug"]
    noise_config = config_all["noise"]

    image_size = config["img_size"]
    task = config["task"]
    scale_factor = config["scale_factor"]
    alpha = config["alpha"]
    NFE = config["NFE"]
    guidance_scale = config["guidance_scale"]
    lr = config["lr"]
    lr_t_ada = config["lr_t_ada"]
    epochs = config["epochs"]
    loss_multiplier = config["loss_multiplier"]
    loss_fn = config["loss_fn"]
    vae_weight = config["vae_weight"]
    lpips_weight = config["lpips_weight"]
    TV_reg_weight = config["TV_reg_weight"]
    prompt = config["prompt"]
    t_end = config["t_end"]
    data_type = eval(config["data_type"])

    # Fix a seed
    set_seed(123)

    # Setup device as cuda
    device = torch.device("cuda")

    # Global variables for wandb
    API_KEY = "bb47140c09574b488dcf0bc3d92f09d93b6241ce"
    PROJECT_NAME = "FMPlug-Ada"

    # Enable wandb
    print("Initialize Project ...")
    wandb.login(key=API_KEY)  # type: ignore

    wandb_instance = wandb.init(  # type: ignore
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=["Experimental", task],
        config={
            "lr": lr,
            "lr_t_ada": lr_t_ada,
            "epochs": epochs,
            "loss_multiplier": loss_multiplier,
            "image_size": image_size,
            "scale_factor": scale_factor,
            "NFE": NFE,
            "guidance_scale": guidance_scale,
            "prompt": prompt,
            "loss_fn": loss_fn,
            "vae_weight": vae_weight,
            "lpips_weight": lpips_weight,
            "TV_reg_weight": TV_reg_weight,
            "t_end": t_end,
            "data_type": data_type
        },
    )

    # Create the directory to save all of the model results
    wandb_experiment_id = wandb_instance.id
    save_file_path = f"/scratch.global/wan01530/FMPlug/experiments/{wandb_experiment_id}"
    os.makedirs(save_file_path, exist_ok=True)

    gt_img_path = "/scratch.global/wan01530/FMPlug/data/div2k_example.png"
    gt_img = Image.open(gt_img_path).convert("RGB")

    tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    gt_img = tf(gt_img)

    ref_numpy = np.array(gt_img)
    x = gt_img * 2.0 - 1.0  # type: ignore

    ref_img = torch.Tensor(x).to(data_type).to(device).unsqueeze(0)
    ref_img.requires_grad = False

    operator_config = {
            "operator": {
                "name": task,
                "in_shape": (1, 3, image_size, image_size),
                "scale_factor": scale_factor,
            }
    }
    # Initalize operator
    operator = get_operator(device=device, **operator_config["operator"])  # type: ignore
    noiser = get_noise(**noise_config)  # type: ignore

    # Forward measurement model (Ax + n)
    y = operator.forward(ref_img)
    y_n = noiser(y)

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
    vae.requires_grad_(False)
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
    

    # initialize latent
    
    y_n = y_n.to(device)
    if "super_resolution" in task:
        img = y_n.reshape([1, -1, image_size//int(config['scale_factor']), image_size//int(config['scale_factor'])])
        img = transforms.functional.resize(img, [image_size, image_size])
    else:
        img = y_n.reshape([1, -1, image_size, image_size])
    img = img.to(y_n.dtype)
    img = img.to(device)
    with torch.no_grad():
        z = encode(img)
        z = np.sqrt(alpha) * z + np.sqrt(1 - alpha) * torch.randn_like(z)
        z = z.detach()
    
    del img

        
    z = torch.nn.parameter.Parameter(z, True).to(device)
    z = z.requires_grad_(True)
    t_ada = torch.tensor((12.0 * (1-alpha)) - 6.0).to(device)
    t_ada = t_ada.requires_grad_(True)

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
        )

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

    params_group1 = {'params': z, 'lr': lr}
    params_group2 = {'params': t_ada, 'lr': lr_t_ada}
    optimizer = torch.optim.AdamW([params_group1, params_group2])
    
    # params = [z, t_ada]
    # optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=50, history_size=50, line_search_fn="strong_wolfe")
    
    decay_factor = 0.95

    psnrs = []
    losses = []
    best_images = []

    early_stopping_counter = 0
    
    # Current memory usage by tensors (in MB)
    print("Init Opt Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
    print("Init Opt Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**2, 2), "MB")
    decoded_output = None
    for iterator in tqdm.tqdm(range(epochs)):
        print("Iter: ", iterator)
        def closure():
            nonlocal decoded_output
            optimizer.zero_grad()

            with (torch.cuda.amp.autocast(enabled=True, dtype=data_type),):
                # x_t = integrate(
                #     f,
                #     z,
                #     t_ada,
                #     NFE,
                #     prompt_embedding,
                #     pooled_embedding,
                #     device,
                #     guidance_scale=guidance_scale,
                # )
                x_t = checkpoint(checkpointed_integrate, z)
                x_t = (x_t / vae.config.scaling_factor) + vae.config.shift_factor
                decoded_output = vae.decode(x_t).sample


            decoded_output = torch.sin(decoded_output)
            operator_decoded_output = operator.forward(decoded_output)

            loss = criterion(operator_decoded_output, y_n)
            loss += vae_weight * criterion(x_t, vae.encode(decoded_output).latent_dist.sample())
            # print("y_n: ", y_n.shape)
            # print("operator_decoded_output: ", operator_decoded_output.shape)
            # print("percep_loss_fn: ", percep_loss_fn(operator_decoded_output, y_n).shape)
            # loss += lpips_weight * percep_loss_fn(operator_decoded_output, y_n).mean()
            loss += lpips_weight * percep_loss_fn((operator_decoded_output + 1.0) / 2.0, (y_n + 1.0) / 2.0)
            loss += TV_reg_weight * L1_tv(decoded_output) / L2_tv(decoded_output) / decoded_output.numel() / 2.0
            loss *= loss_multiplier

            loss.backward()

            # print out gradients so we can understand
            print("gradient z: ", z.grad.min(), z.grad.max())
            new_lr = lr_t_ada * (decay_factor ** iterator)
            optimizer.param_groups[1]['lr'] = new_lr

            del x_t
            torch.cuda.empty_cache()

            return loss


        loss = optimizer.step(closure)
        # scheduler.step()

        losses.append(loss.item())
        
        print("Opt Allocated:", round(torch.cuda.memory_allocated(0) / 1024**2, 2), "MB")
        print("Opt Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**2, 2), "MB")

        # Evaluate
        with torch.no_grad():
            output = decoded_output.detach()
            lpips_score = lpips_loss_fn(output, ref_img).mean()
            output_numpy = np.clip((output.cpu().squeeze().numpy() + 1) / 2, 0, 1)
            output_numpy = np.transpose(
                output_numpy, (1, 2, 0)
            )  # Keep out for now lets evaluate

            # calculate psnr
            tmp_psnr = peak_signal_noise_ratio(
                ref_numpy.transpose(1, 2, 0), output_numpy
            )
            print(tmp_psnr)

            # calculate mse
            mse_score = ((ref_numpy.transpose(1, 2, 0) - output_numpy) ** 2).mean()

            metrics_to_log = {
                "epoch": iterator,
                "t_ada": (1000 * torch.sigmoid(t_ada)).item(),
                "loss": loss.item()/loss_multiplier,
                "psnr": tmp_psnr,
                "lpips": lpips_score.item(),
                "mse_loss": mse_score,
            }
            wandb.log(metrics_to_log)  # type: ignore

            psnrs.append(tmp_psnr)

            if len(psnrs) == 1 or (len(psnrs) > 1 and tmp_psnr > np.max(psnrs[:-1])):
                best_img = output_numpy
                best_images.append(best_img)
                # early_stopping_counter = 0

            # else:
            #     early_stopping_counter += 1

            # if early_stopping_counter >= 1:
            #     break

    display_ref_img = ref_numpy.transpose(1, 2, 0)

    # Create a figure with 1 row, 3 columns
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 10))

    # Ground truth
    ax1.imshow(display_ref_img)
    ax1.set_title("Ground Truth Image")
    ax1.axis("off")

    # Display the corrupted image
    y_n_numpy = y_n.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    y_n_numpy = (y_n_numpy + 1.0) / 2.0
    ax2.imshow(y_n_numpy)
    ax2.set_title("Corrupted Image")
    ax2.axis("off")

    # Reconstructed image
    ax3.imshow(best_img.astype(float))
    ax3.set_title("Reconstructed Image (FM)")
    ax3.axis("off")

    # Pixel-wise absolute difference (grayscale)
    diff = np.abs(display_ref_img - best_img).astype(float).mean(axis=-1)
    ax4.imshow(diff, cmap="hot")
    ax4.set_title("Pixel Difference (FM)")
    ax4.axis("off")

    # Save the figure
    plt.tight_layout()
    fig.savefig(
        os.path.join(save_file_path, f"{wandb_experiment_id}_best_img.png"),
        bbox_inches="tight",
    )
    plt.close()

    # Save the raw data as well
    np.save(os.path.join(save_file_path, "ground_truth_image.npy"), display_ref_img)
    np.save(os.path.join(save_file_path, "reconstructed_image.npy"), best_img)
    # Define where to save
    os.makedirs(save_file_path, exist_ok=True)  # create the folder if it does not exist

    # Save the file
    config_filename = os.path.join(save_file_path, "config.yaml")
    with open(config_filename, "w") as file:
        yaml.safe_dump(config_all, file, default_flow_style=False)
