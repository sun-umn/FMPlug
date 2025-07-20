# stdlib
import os

# third party
# third-party
import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
import wandb
import yaml  # type: ignore
from skimage.metrics import peak_signal_noise_ratio

# first party
from fmplug.models.stable_diffusion import StableDiffusion3BaseV2
from fmplug.ode_solver.euler import integrate_euler_v2
from fmplug.tasks.utils import prepare_super_resolution_measurement
from fmplug.utils.measurements import get_noise, get_operator


def sd3_regression_task(config_name: str) -> None:
    """
    Function to run SD3 regression task. This includes the full
    flow matching process and the vae decoder.
    """
    base_config_path = "/users/5/dever120/FMPlug/fmplug/configs"
    with open(os.path.join(base_config_path, config_name + ".yaml"), "r") as file:
        config = yaml.safe_load(file)

    # Setup device as cuda
    device = torch.device("cuda")

    # Image & training config
    image_size = config["config"].get("image_size")
    scale_factor = config["config"].get("scale_factor")
    epochs = config["config"].get("epochs")

    # SD3 configuration
    lr = config["config"].get("lr")
    num_inference_steps = config["config"].get("num_inference_steps")
    guidance_scale = config["config"].get("guidance_scale")
    strength = config["config"].get("strength")

    # Global variables for wandb
    API_KEY = os.environ.get("WANDB_API_KEY")
    PROJECT_NAME = "FMPlug Discover"

    # Enable wandb
    print("Initialize Project ...")
    wandb.login(key=API_KEY)  # type: ignore

    wandb_instance = wandb.init(  # type: ignore
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=["Experimental", "SD3 Regression"],
        config={
            "lr": lr,
            "epochs": epochs,
            "image_size": image_size,
            "scale_factor": scale_factor,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "config_name": config_name,
        },
    )

    # Create the directory to save all of the model results
    wandb_experiment_id = wandb_instance.id
    save_file_path = f"/users/5/dever120/FMPlug/experiments/{wandb_experiment_id}"
    os.makedirs(save_file_path, exist_ok=True)

    # Load an image and simulate the measurement process
    noise_sigma = 0.03
    img_outputs = prepare_super_resolution_measurement(
        image_path="/users/5/dever120/FMPlug/data/afhq_cat.png",
        image_size=image_size,
        scale_factor=scale_factor,
        noise_sigma=noise_sigma,
        device=device,
        get_operator_fn=get_operator,
        get_noise_fn=get_noise,
    )

    print("Load in SD3 image to image model ...")
    sd3_pipeline = StableDiffusion3BaseV2(
        model_key="stabilityai/stable-diffusion-3-medium-diffusers",
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        device=device,
    )

    prompt = "a calico cat with green eyes in front of a white background"

    # Get prompt and pooled embeds
    prompt_embeds, pooled_prompt_embeds = sd3_pipeline.encode_prompts(
        prompt=prompt,
    )

    # Collect timesteps and sigmas
    timesteps, sigmas, _ = sd3_pipeline.retrieve_timesteps_and_sigmas(strength=strength)

    ref_img = img_outputs["ref_img"].squeeze(0).permute(1, 2, 0).cpu()
    ref_img = (ref_img + 1.0) / 2.0
    ref_img = ref_img.to(device)

    # This is where we will handle different strengths
    # The vae encoder expects an image in the ranage [-1, 1]. Evidence
    # is proved by image_processor.preprocess
    # If the strength is 1.0 then we start from a random guassian
    # distribution
    if strength == 1.0:
        latents = torch.load("latents.pt")

        # z is the value that we will optimize
        z = sd3_pipeline.prepare_latents(latents=latents)

    else:
        print("Strength < 1.0 logic ...")
        # Otherwise the strength is a linear combination of the
        # latent and random gaussian
        # For these tasks we will use noisy GT as the input
        # to the encoder
        img_to_encode = (ref_img * 2.0) - 1.0

        # Add a small amount of noise to the GT image
        img_to_encode = ref_img + noise_sigma * torch.randn(
            ref_img.shape, device=device, dtype=torch.float32
        )
        img_to_encode = img_to_encode.permute(2, 0, 1).unsqueeze(0)

        latents = sd3_pipeline.vae.encode(img_to_encode).latent_dist.sample()

        # Then we scale and shift the encoding
        latents = (
            latents - sd3_pipeline.vae.config.shift_factor
        ) * sd3_pipeline.vae.config.scaling_factor

        # Finally, we add noise based on a linear combination of
        # random noise and the latent variable
        sigma = sigmas[0]
        noise = torch.randn(
            latents.shape, generator=None, device=device, dtype=torch.float32
        )

        latents = sigma * noise + (1.0 - sigma) * latents
        z = latents

    z = z.requires_grad_(True)

    # We will log this with wandb
    initial_z_min = z.min()
    initial_z_max = z.max()
    initial_z_mean = z.mean()

    # Configure optimization
    psnr_scores_list = []
    mse_scores_list = []

    optimizer = torch.optim.Adam([z], lr=lr)

    for epoch in tqdm.tqdm(range(epochs)):
        optimizer.zero_grad()

        x_t = integrate_euler_v2(
            f=sd3_pipeline.predict,
            x0=z,
            timesteps=timesteps,
            sigmas=sigmas,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance_scale=guidance_scale,
        )

        # What if we try with no shifting or scaling
        x_t = (
            x_t / sd3_pipeline.vae.config.scaling_factor
        ) + sd3_pipeline.vae.config.shift_factor

        # This code comes directly from the SD3 pipeline
        decoded_img = sd3_pipeline.vae.decode(x_t, return_dict=False)[0]
        decoded_img = decoded_img.squeeze(0).permute(1, 2, 0)
        decoded_img = (decoded_img * 0.5) + 0.5
        decoded_img = torch.clamp(decoded_img, 0, 1)

        loss = ((ref_img - decoded_img) ** 2).mean()

        loss.backward()
        optimizer.step()

        # Get the z grad value
        grad_norm = z.grad.norm()  # type: ignore

        with torch.no_grad():
            psnr_score = peak_signal_noise_ratio(
                ref_img.cpu().numpy(), decoded_img.detach().cpu().numpy()
            )
            psnr_scores_list.append(psnr_score)
            mse_scores_list.append(loss.item())

            metrics_to_log = {
                "epoch": epoch,
                "mse_loss": loss.item(),
                "psnr": psnr_score,
                "grad_norm": grad_norm,
                "initial_z_min": initial_z_min,
                "initial_z_mean": initial_z_mean,
                "initial_z_max": initial_z_max,
                "z_min": z.min(),
                "z_mean": z.mean(),
                "z_max": z.max(),
            }
            wandb.log(metrics_to_log)  # type: ignore

    # Save the original image and the reconstruction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(ref_img.cpu().numpy().astype(float))  # type: ignore
    ax1.set_title("Original Image")

    ax2.imshow(decoded_img.detach().cpu().numpy().astype(float))  # type: ignore
    ax2.set_title("Reconstructed Image")

    image_save_path = os.path.join(save_file_path, "final_output.png")
    fig.savefig(image_save_path, bbox_inches="tight")

    # Save the original image and its historgram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(ref_img.cpu().numpy().astype(float))  # type: ignore
    ax1.set_title("Original Image")

    pd.Series(ref_img.detach().cpu().numpy().astype(float).flatten()).hist(
        bins=100, ax=ax2
    )  # type: ignore  # noqa
    ax2.set_title("Original Image Histogram")

    image_save_path = os.path.join(save_file_path, "ref_image_output.png")
    fig.savefig(image_save_path, bbox_inches="tight")

    # Save the original z distribution vs the final distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    pd.Series(latents.detach().cpu().numpy().astype(float).flatten()).hist(
        bins=100, ax=ax1
    )  # type: ignore  # noqa
    ax1.set_title("Initial z Distribution")

    pd.Series(z.detach().cpu().numpy().astype(float).flatten()).hist(bins=100, ax=ax2)  # type: ignore  # noqa
    ax2.set_title("Final z Distribution")

    image_save_path = os.path.join(save_file_path, "z_distribution_output.png")
    fig.savefig(image_save_path, bbox_inches="tight")

    # Save the final decoded image and its histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(decoded_img.detach().cpu().numpy().astype(float))  # type: ignore
    ax1.set_title("Trained Decoded Image")

    pd.Series(decoded_img.detach().cpu().numpy().astype(float).flatten()).hist(
        bins=100, ax=ax2
    )
    ax2.set_title("Trained Decoded Image Histogram")

    image_save_path = os.path.join(save_file_path, "trained_decoded_image_output.png")
    fig.savefig(image_save_path, bbox_inches="tight")

    wandb.finish()  # type: ignore
