# stdlib
import os
import random
import time

# third party
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.schedulers.scheduling_flow_match_heun_discrete import (
    FlowMatchHeunDiscreteScheduler,
)
from skimage.metrics import peak_signal_noise_ratio
from torchmetrics.image import StructuralSimilarityIndexMeasure

import wandb

# first party
from fmplug.models.stable_diffusion import StableDiffusion3BaseV2
from fmplug.ode_solver.euler import integrate_euler
from fmplug.ode_solver.heun2 import integrate_heun
from fmplug.tasks.utils import compute_ssim, prepare_super_resolution_measurement
from fmplug.utils.measurements import get_noise, get_operator

# These presets are used for torch compile for SD3
torch.set_float32_matmul_precision("high")

# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True


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


def total_variation_loss(x):
    return torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.mean(
        torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    )


def super_resolution_task(config_name: str) -> None:
    # Configuration
    image_size = 512
    scale_factor = 4
    num_inference_steps = 12
    guidance_scale = 3.0
    lr = 1e-2
    epochs = 1000
    strength = 0.90
    ode_solver = "euler"

    # NOTE: Seed was 123
    device = torch.device("cuda")
    set_seed(0)

    # Global variables for wandb
    API_KEY = os.environ.get("WANDB_API_KEY")
    PROJECT_NAME = "FMPlug"

    # Enable wandb
    print("Initialize Project ...")
    wandb.login(key=API_KEY)  # type: ignore

    wandb_instance = wandb.init(  # type: ignore
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=["Experimental", "Super Resolution"],
        config={
            "lr": lr,
            "epochs": epochs,
            "image_size": image_size,
            "scale_factor": scale_factor,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "config_name": config_name,
            "ode_solver": ode_solver,
        },
    )

    # Create the directory to save all of the model results
    wandb_experiment_id = wandb_instance.id
    save_file_path = f"/users/5/dever120/FMPlug/experiments/{wandb_experiment_id}"
    os.makedirs(save_file_path, exist_ok=True)

    # Load an image and simulate the measurement process
    noise_sigma = 0.03
    img_outputs = prepare_super_resolution_measurement(
        image_path="/users/5/dever120/FMPlug/data/ffhq_baby.png",
        # image_path="/users/5/dever120/FMPlug/data/afhq_cat.png",
        # image_path="/users/5/dever120/FMPlug/data/00000-baby.png",
        image_size=image_size,
        scale_factor=scale_factor,
        noise_sigma=noise_sigma,
        device=device,
        get_operator_fn=get_operator,
        get_noise_fn=get_noise,
    )

    print("Load in SD3 image to image model ...")
    if ode_solver == "heun":
        sd3_pipeline = StableDiffusion3BaseV2(
            model_key="stabilityai/stable-diffusion-3-medium-diffusers",
            scheduler=FlowMatchHeunDiscreteScheduler(),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )

    elif ode_solver == "euler":
        sd3_pipeline = StableDiffusion3BaseV2(
            model_key="stabilityai/stable-diffusion-3-medium-diffusers",
            scheduler=FlowMatchEulerDiscreteScheduler(shift=4.0),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )

    prompt = """
        A close-up portrait of a baby with soft skin, short dark hair,
        and calm expression, wearing a green onesie, resting against
        a turquoise cushion.
    """

    prompt_2 = """
        Ultra-realistic portrait, natural lighting, soft shadows,
        sharp facial features, high-resolution skin texture,
        lifelike color tones, DSLR photo.
    """

    # Get prompt and pooled embeds
    prompt_embeds, pooled_prompt_embeds = sd3_pipeline.encode_prompts(
        prompt=prompt,
        # prompt_2=prompt_2,
    )

    # Collect timesteps and sigmas
    print(sd3_pipeline.scheduler.sigmas)
    timesteps, sigmas, _ = sd3_pipeline.retrieve_timesteps_and_sigmas(strength=strength)

    # We need to pass a step index for Heun 2
    sd3_pipeline.scheduler._init_step_index(timesteps[0])

    # Step index need to be properly set to have good quality images
    step_index = sd3_pipeline.scheduler._step_index

    # Reference image to view
    ref_img = img_outputs["ref_img"].squeeze(0).permute(1, 2, 0).cpu()
    ref_img = (ref_img + 1.0) / 2.0
    ref_img = ref_img.to(device)

    # Get the degraded image
    y_n = img_outputs["y_n"]

    # This is where we will handle different strengths
    # The vae encoder expects an image in the ranage [-1, 1]. Evidence
    # is proved by image_processor.preprocess
    # If the strength is 1.0 then we start from a random guassian
    # distribution
    if strength == 1.0:
        # When strength is 1 then we have a standard normal guassian
        # this is similar to text-to-image
        z = sd3_pipeline.prepare_latents(latents=None)

    else:
        print("Strength < 1.0 logic ...")
        # Otherwise the strength is a linear combination of the
        # latent and random gaussian
        # For these tasks we will use noisy GT as the input
        # to the encoder
        # NOTE: This initialization is for super resolution
        img_to_encode = torch.nn.functional.interpolate(
            y_n,  # y_n is already in the range [-1, 1]
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

        # Encoder expects images in -1 to 1 so just clamp to make sure
        # this is true
        img_to_encode = torch.clamp(img_to_encode, -1.0, 1.0)
        latents = sd3_pipeline.vae.encode(img_to_encode).latent_dist.sample()

        # Then we scale and shift the encoding
        latents = (
            latents - sd3_pipeline.vae.config.shift_factor
        ) * sd3_pipeline.vae.config.scaling_factor

        # Finally, we add noise based on a linear combination of
        # random noise and the latent variable
        sigma = sigmas[step_index]
        noise = torch.randn(
            latents.shape, generator=None, device=device, dtype=torch.float32
        )

        # This definition comes straight from SD3 pipeline
        latents = sigma * noise + (1.0 - sigma) * latents

        # Initialization from our paper
        latents = latents / latents.norm() * torch.sqrt(torch.tensor(latents.numel()))
        z = latents

    # Make z trainable
    z = z.requires_grad_(True)

    # We will log this with wandb
    initial_z_min = z.min()
    initial_z_max = z.max()
    initial_z_mean = z.mean()

    optimizer = torch.optim.Adam(
        [
            {"params": [z], "lr": lr},
        ]
    )

    # Export the measurment operator
    operator = img_outputs["operator"]

    # Create the loss function
    criterion = torch.nn.L1Loss()
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)

    # Try the grad scaler for fp16
    scaler = torch.amp.GradScaler()

    start = time.time()
    for idx, epoch in tqdm.tqdm(enumerate(range(epochs))):
        torch.compiler.cudagraph_mark_step_begin()

        optimizer.zero_grad()

        # The transformer at least expects a zero mean
        # as an input but can handle standard and non-standard
        # gaussian distributions
        z0 = z - z.mean()

        if ode_solver == "heun":
            x_t = integrate_heun(
                f=sd3_pipeline.predict,
                x0=z0,
                timesteps=timesteps,
                sigmas=sigmas,
                step_index=step_index,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=guidance_scale,
            )

        elif ode_solver == "euler":
            x_t = integrate_euler(
                f=sd3_pipeline.predict,
                x0=z0,
                timesteps=timesteps,
                sigmas=sigmas,
                step_index=step_index,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=guidance_scale,
            )

        # What if we try with no shifting or scaling
        x_t = (
            x_t / sd3_pipeline.vae.config.scaling_factor
        ) + sd3_pipeline.vae.config.shift_factor

        # This code comes directly from the SD3 pipeline
        # This output is from [-1, 1]
        decoded_img = torch.clamp(
            sd3_pipeline.vae.decode(x_t, return_dict=False)[0], -1.0, 1.0
        )

        # Now apply the degradation
        operator_decoded_output = operator.forward(decoded_img)  # type: ignore

        # Apply the loss function - this expects [-1, 1]
        loss = criterion(operator_decoded_output, y_n)

        # Update gradients of z
        scaler.scale(loss).backward()

        # Unscale gradients before clipping
        scaler.unscale_(optimizer)

        # Clip gradients (example: max norm = 1.0)
        # Found this value to work well for Heun2 - may need to be tuned
        # for euler
        torch.nn.utils.clip_grad_norm_([z], max_norm=0.005)

        scaler.step(optimizer)
        scaler.update()

        # What is the gradient norm?
        # Compute the grad norm so we can track it
        grad_norm = z.grad.norm()  # type: ignore

        # Compute the PSNR & print loss
        with torch.no_grad():
            lpips_score = lpips_loss_fn(decoded_img, img_outputs.get("ref_img"))

            decoded_img = decoded_img.squeeze(0)
            model_img = (decoded_img + 1.0) / 2.0  # type: ignore
            model_img = model_img.squeeze(0).detach().cpu().numpy()  # type: ignore

            img = img_outputs.get("ref_img")
            img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)  # type: ignore
            img = img.squeeze(0).detach().cpu().numpy()  # type: ignore

            ssim_score = compute_ssim(
                img,
                model_img,
            )

            img = img.transpose(1, 2, 0)  # type: ignore
            model_img = model_img.transpose(1, 2, 0)  # type: ignore

            psnr_score = peak_signal_noise_ratio(img, model_img, data_range=1.0)
            mse_score = ((img - model_img) ** 2).mean()

            # Log all metrics here and track how z is
            # changing over time
            metrics_to_log = {
                "epoch": epoch,
                "loss": loss.item(),
                "mse_loss": mse_score,
                "psnr": psnr_score,
                "ssim": ssim_score,
                "lpips": lpips_score.item(),
                "grad_norm": grad_norm,
                "initial_z_min": initial_z_min,
                "initial_z_mean": initial_z_mean,
                "initial_z_max": initial_z_max,
                "z_min": z.min(),
                "z_mean": z.mean(),
                "z_max": z.max(),
            }
            wandb.log(metrics_to_log)  # type: ignore

            # Always log the first image to identify any issues
            # Tracking images during training can also help us understand
            # if we leave the manifold of natural images
            if idx == 0:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

                ax1.imshow(img)
                ax1.set_title("Original Image")
                ax1.axis("off")

                ax2.imshow(model_img)
                ax2.set_title("Reconstructed Image")
                ax2.axis("off")

                image_diff = np.abs(img - model_img).mean(axis=-1)
                ax3.imshow(image_diff)
                ax3.set_title("Image Differential")
                ax3.axis("off")

                image_save_path = os.path.join(
                    save_file_path, f"reference_vs_generated_image_epoch_{idx + 1}.png"
                )

                fig.tight_layout()
                fig.savefig(image_save_path, bbox_inches="tight")
                plt.close(fig)

    end = time.time()
    print(end - start)

    # The prompt is playing a major role in how good of a solution we can obtain
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5, 15))

    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(model_img)
    ax2.set_title("Reconstructed Image")
    ax2.axis("off")

    image_diff = np.abs(img - model_img).mean(axis=-1)
    ax3.imshow(image_diff)
    ax3.set_title("Pixel Difference")
    ax3.axis("off")

    image_save_path = os.path.join(save_file_path, "final_output.png")
    fig.tight_layout()
    fig.savefig(image_save_path, bbox_inches="tight")

    wandb.finish()  # type: ignore
