# stdlib
import inspect
import os
import random
from typing import List, Optional, Union

# third party
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
import yaml  # type: ignore
from diffusers import StableDiffusion3Img2ImgPipeline
from huggingface_hub import login
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from torchvision import transforms

# first party
from fmplug.layers.activations import CoordFeatureSiren
from fmplug.losses.losses import PerceptualLossV3
from fmplug.utils.measurements import get_noise, get_operator


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


def normalize_z(z, target_norm=400.0, eps=1e-8):
    # z: (B, C, H, W)
    current_norm = torch.norm(z.view(z.shape[0], -1), dim=1, keepdim=True)  # (B, 1)
    current_norm = current_norm.view(-1, 1, 1, 1)
    scale = target_norm / (current_norm + eps)
    return z * scale


def normalize_mean_std(z, eps=1e-8):
    """
    Normalize each sample in z to have zero mean and unit variance.

    Args:
        z (torch.Tensor): Shape (B, C, H, W)
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Normalized tensor with mean=0 and std=1 per sample.
    """
    mean = z.mean(dim=(1, 2, 3), keepdim=True)
    std = z.std(dim=(1, 2, 3), keepdim=True)
    return (z - mean) / (std + eps)


def normalize_to(z, target_mean=0.0, target_std=1.0, eps=1e-8):
    z_normed = normalize_mean_std(z, eps)
    return z_normed * target_std + target_mean


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


def integrate(
    f,
    x0,
    timesteps,
    sigmas,
    prompt_embedding,
    pooled_embedding,
    device,
    guidance_scale: float = 7.0,
):
    # Start ODE solver
    current_timesteps = timesteps[:-1]
    previous_timesteps = timesteps[1:]
    current_sigmas = sigmas[:-1]
    previous_sigmas = sigmas[1:]

    integrate_parameters = zip(
        current_timesteps,
        previous_timesteps,
        current_sigmas,
        previous_sigmas,
    )

    do_classifier_free_guidance = guidance_scale > 1.0

    for i, (t0, t1, sigma, sigma_next) in enumerate(integrate_parameters):
        # print(x0.norm(), x0.mean(), x0.var())
        # x0 will be the latent variable
        latent_model_input = torch.cat([x0] * 2) if do_classifier_free_guidance else x0

        # broadcast to batch dimension in a way that's compatible with ONNX / Core ML
        timestep = t0.expand(latent_model_input.shape[0])
        prev_timestep = t1.expand(latent_model_input.shape[0])

        # upcast to avoid precision issues
        sample = x0.to(torch.float32)
        # target_norm = sample.norm()
        # target_mean = sample.mean()
        # target_std = sample.std()
        dt = sigma_next - sigma

        # # Euler
        # noise_pred = f(
        #     x=latent_model_input,
        #     t=timestep,
        #     prompt_embedding=prompt_embedding,
        #     pooled_embedding=pooled_embedding,
        #     device=device,
        # )

        # Heun2
        k1 = f(
            x=latent_model_input,
            t=timestep,
            prompt_embedding=prompt_embedding,
            pooled_embedding=pooled_embedding,
            device=device,
        )

        # Predict next latent using Euler step
        x1_pred = latent_model_input + dt * k1

        # k2
        k2 = f(
            x=x1_pred,
            t=prev_timestep,
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
        prev_sample = sample + noise_pred

        prev_sample = prev_sample.to(torch.float32)
        # prev_sample = normalize_z(prev_sample, target_norm=target_norm)
        # prev_sample = normalize_to(prev_sample, target_mean=target_mean, target_std=target_std, eps=1e-8)  # noqa

        # NOTE: There was a bug in the notebook and solution is very blurry
        # and does not provide a good prior. Commenting out for now.
        # while j < len(timesteps) and t1 >= timesteps[j]:
        #     solution = linear_interp(
        #         sigma, sigma_next, sample, prev_sample, timesteps[j]
        #     )
        #     j += 1

        x0 = prev_sample

    return x0


def super_resolution_task(config_name: str) -> None:
    # Load in the configuration
    base_config_path = "/users/5/dever120/FMPlug/fmplug/configs"
    with open(os.path.join(base_config_path, config_name + ".yaml"), "r") as file:
        config = yaml.safe_load(file)["config"]

    image_size = height = width = config["image_size"]
    scale_factor = config["scale_factor"]
    num_inference_steps = config["num_inference_steps"]
    batch_size = config["batch_size"]
    guidance_scale = config["guidance_scale"]
    lr = config["lr"]
    epochs = config["epochs"]
    loss_multiplier = config["loss_multiplier"]
    loss_fn = config["loss_fn"]
    max_iter = config["max_iter"]
    prompt = config["prompt"]
    timesteps_type = config["timesteps_type"]
    t_start = config["t_start"]
    t_end = config["t_end"]

    # Fix a seed
    set_seed(2009)

    # Log into huggingface to be able to pull the SD3.0
    print("Log into HuggingFace ...")
    login("REMOVED")

    # Setup device as cuda
    device = torch.device("cuda")

    # Global variables for wandb
    API_KEY = "2080070c4753d0384b073105ed75e1f46669e4bf"
    PROJECT_NAME = "FMPlug"

    # Enable wandb
    print("Initialize Project ...")
    wandb.login(key=API_KEY)  # type: ignore

    wandb_instance = wandb.init(  # type: ignore
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=["Experimental", "super resolution"],
        config={
            "lr": lr,
            "epochs": epochs,
            "loss_multiplier": loss_multiplier,
            "image_size": image_size,
            "scale_factor": scale_factor,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "lbfgs_max_iter": max_iter,
            "prompt": prompt,
            "loss_fn": loss_fn,
            "timesteps_type": timesteps_type,
            "t_start": t_start,
            "t_end": t_end,
        },
    )

    # Create the directory to save all of the model results
    wandb_experiment_id = wandb_instance.id
    save_file_path = f"/users/5/dever120/FMPlug/experiments/{wandb_experiment_id}"
    os.makedirs(save_file_path, exist_ok=True)

    gt_img_path = "/users/5/dever120/FMPlug/data/div2k_example.png"
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

    ref_img = torch.Tensor(x).to(torch.float32).to(device).unsqueeze(0)
    ref_img.requires_grad = False

    # Super resolution
    # TODO: make image size configurable
    config = {
        "measurement": {
            "operator": {
                "name": "super_resolution",
                "in_shape": (1, 3, image_size, image_size),
                "scale_factor": scale_factor,
            },
            "noise": {"name": "gaussian", "sigma": 0.01},
        },
    }

    measure_config = config["measurement"]

    # Initalize operator
    operator = get_operator(device=device, **measure_config["operator"])  # type: ignore
    noiser = get_noise(**measure_config["noise"])  # type: ignore

    # Forward measurement model (Ax + n)
    y = operator.forward(ref_img)
    y_n = noiser(y)

    print("Load in SD3 image to image model ...")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=torch.float32,
    )
    pipe = pipe.to(device)
    # pipe.enable_model_cpu_offload()

    # Extract different components of the pipeline
    prompt_encoder = pipe.encode_prompt
    image_processor = pipe.image_processor
    prepare_latents = pipe.prepare_latents

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

    # Start the problem by defining z
    # The inputs for the image to image pipeline need to be between
    # 0 and 1
    z = torch.rand(3, image_size, image_size)

    # prompt
    # prompt = "a high quality, unclose photo of a red panda's face in the jungle"
    # prompt_2 = (
    #     "Ultra-detailed photo of a red panda in the wild, realistic fur, "
    #     "bright expressive eyes, eating bamboo leaves, DSLR wildlife photography, "
    #     "bokeh background"
    # )
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

    # preprocess image
    print("Process image, build timesteps & latent variables ...")
    image = image_processor.preprocess(z, height=height, width=width)

    # 4. Prepare timesteps
    strength = 0.95
    mu = None
    sigmas = None

    scheduler_kwargs = {}
    if pipe.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        image_seq_len = (
            int(height) // pipe.vae_scale_factor // transformer.config.patch_size
        ) * (int(width) // pipe.vae_scale_factor // transformer.config.patch_size)
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.16),
        )
        scheduler_kwargs["mu"] = mu

    elif mu is not None:
        scheduler_kwargs["mu"] = mu
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, sigmas=sigmas, **scheduler_kwargs
    )

    timesteps, num_inference_steps = pipe.get_timesteps(
        num_inference_steps, strength, device
    )

    # timesteps = torch.linspace(1000.0, 0.0, num_inference_steps + 1).to(device)
    if timesteps_type == "exponential":
        timesteps = make_exponential_timesteps(
            t_start=t_start, t_end=t_end, num_steps=(num_inference_steps + 1), decay=5.0
        )
    elif timesteps_type == "linear":
        timesteps = torch.linspace(t_start, t_end, num_inference_steps + 1).to(device)

    timesteps = timesteps.to(device)
    sigmas = timesteps / 1000.0
    print(timesteps, sigmas)

    # Prepare the latent timestep so we can build the latent variable
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

    # 5. Prepare latent variables
    latents = prepare_latents(
        image,
        latent_timestep,
        batch_size,
        num_images_per_prompt,
        prompt_embeds.dtype,
        device,
        generator=None,
    )

    # NOTE: What happens if this is just random?
    # latents = torch.randn((1, 16, 96, 96)).to(device=device, dtype=torch.float32)
    z = torch.nn.parameter.Parameter(latents)

    # amplitude = torch.tensor(torch.pi).to(dtype=torch.float32, device=device)
    timesteps = torch.nn.parameter.Parameter(timesteps)

    # Solve the ODE
    print("Solve inverse problem ...")

    def f(x, t, prompt_embedding, pooled_embedding, device):
        with torch.amp.autocast(device.type, dtype=torch.float32):
            # result = vae.decode(x).sample
            # result = vae.encode(result).latent_dist.sample()

            result = transformer(
                hidden_states=x,
                timestep=t,
                encoder_hidden_states=prompt_embedding,
                pooled_projections=pooled_embedding,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        return result

    # Criterion for learning
    if loss_fn == "l1":
        criterion = torch.nn.L1Loss().to(device)

    elif loss_fn == "mse":
        criterion = torch.nn.MSELoss().to(device)  # type: ignore

    # Setup perceptual loss
    # percep_loss_fn = PerceptualLoss(layers=["relu1_2"]).to(device)
    percep_loss_fn = PerceptualLossV3(layers=["relu1_2"]).to(device)
    # style_loss_fn = StyleLoss(layers=["relu1_2"]).to(device)
    # fft_loss_fn = FFTLoss()

    # Initialize sine activation
    _ = CoordFeatureSiren(
        feature_dim=3,
        hidden_dim=3,
        out_dim=3,
        depth=3,
        omega_0=30.0,
    ).to(device)

    # parameter groups for LBFGS
    params_group = {"params": [z], "lr": lr}

    optimizer = torch.optim.LBFGS(
        [params_group],
        max_iter=max_iter,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    def closure():
        # closure_timesteps = torch.clamp(timesteps, 1e-5, 1000.0)
        # closure_sigmas = closure_timesteps / 1000.0

        optimizer.zero_grad()

        with (torch.cuda.amp.autocast(enabled=True, dtype=torch.float32),):
            x_t = integrate(
                f,
                z,
                timesteps,
                sigmas,
                # closure_timesteps,
                # closure_sigmas,
                prompt_embedding,
                pooled_embedding,
                device,
                guidance_scale=guidance_scale,
            )
            x_t = (x_t / vae.config.scaling_factor) + vae.config.shift_factor
            decoded_output = vae.decode(x_t.half()).sample
            # decoded_output = image_processor.denormalize(decoded_output)
            # decoded_output = (decoded_output * 2.0) - 1.0  # Need to put it in the range of -1 to 1  # noqa
            # decoded_output = vae.decode(x_t)

        # decoded_output = decoded_output.to(torch.float32)
        decoded_output = torch.sin(decoded_output.to(torch.float32))
        decoded_output = decoded_output.to(torch.float32)

        # amplitude = torch.clamp(amplitude, min=0.5)
        # decoded_output = torch.sin(decoded_output / 2.0)
        # decoded_output = sine_layer(decoded_output)
        # decoded_output = torch.clamp(decoded_output, -1.0, 1.0)

        # Output shapes checkout
        # Verified that outputs are between -1 and 1
        # print(decoded_output.min(), decoded_output.max())

        # Store for access after step
        decoded_output_holder[0] = decoded_output.detach()

        operator_decoded_output = operator.forward(decoded_output)

        loss = loss_multiplier * criterion(operator_decoded_output, y_n)
        # loss = loss_multiplier * criterion(operator_decoded_output, y_input)
        loss += loss_multiplier * percep_loss_fn(
            (operator_decoded_output + 1.0) / 2.0, (y_n + 1.0) / 2.0
        )
        # loss += loss_multiplier * style_loss_fn((operator_decoded_output + 1.0) / 2.0, (y_n + 1.0) / 2.0)  # noqa
        # loss += loss_multiplier * fft_loss_fn(operator_decoded_output, y_n)

        loss.backward()

        # print out gradients so we can understand
        print(z.grad.min(), z.grad.max())

        del x_t
        torch.cuda.empty_cache()

        return loss

    psnrs = []
    losses = []
    best_images = []

    early_stopping_counter = 0
    for iterator in tqdm.tqdm(range(epochs)):
        decoded_output_holder = [None]  # mutable object to hold output

        loss = optimizer.step(closure)
        # scheduler.step()

        losses.append(loss.item())

        with torch.no_grad():
            x_t = integrate(
                f,
                z,
                timesteps,
                sigmas,
                prompt_embedding,
                pooled_embedding,
                device,
                guidance_scale=guidance_scale,
            )
            x_t = (x_t / vae.config.scaling_factor) + vae.config.shift_factor
            decoded_output = vae.decode(x_t.to(torch.float32)).sample
            decoded_output = torch.sin(decoded_output)
            # decoded_output = image_processor.denormalize(decoded_output) # Puts in 0 to 1  # noqa

        decoded_output = decoded_output.to(torch.float32)

        # amplitude = torch.clamp(amplitude, min=0.5)
        # decoded_output = torch.sin(decoded_output / 2.0)
        # decoded_output = sine_layer(decoded_output)
        # decoded_output = torch.clamp(decoded_output, -1.0, 1.0)
        # sigmas = timesteps / 1000.0
        print(timesteps)
        print(sigmas)

        # Evaluate
        with torch.no_grad():
            output_numpy = decoded_output.detach().cpu().squeeze().numpy()
            output_numpy = np.clip((output_numpy + 1) / 2, 0, 1)
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
                "psnr": tmp_psnr,
                "mse_loss": mse_score,
            }
            wandb.log(metrics_to_log)  # type: ignore

            psnrs.append(tmp_psnr)

            if len(psnrs) == 1 or (len(psnrs) > 1 and tmp_psnr > np.max(psnrs[:-1])):
                best_img = output_numpy
                best_images.append(best_img)
                early_stopping_counter = 0

            else:
                early_stopping_counter += 1

            if early_stopping_counter >= 1:
                break

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
