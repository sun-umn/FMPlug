# stdlib
import os
import random
import time

# third party
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import tqdm
import wandb
from skimage import exposure
from skimage.metrics import peak_signal_noise_ratio

# first party
from fmplug.models.stable_diffusion import StableDiffusion3BaseV2
from fmplug.ode_solver.euler import integrate_euler_v2
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


def gradient_magnitude_loss(pred, target):
    # Sobel filters
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
    ).view(1, 1, 3, 3)

    # Apply to each channel
    pred_grad_x = F.conv2d(
        pred,
        sobel_x.repeat(pred.shape[1], 1, 1, 1).to(pred.device),
        padding=1,
        groups=pred.shape[1],
    )
    pred_grad_y = F.conv2d(
        pred,
        sobel_y.repeat(pred.shape[1], 1, 1, 1).to(pred.device),
        padding=1,
        groups=pred.shape[1],
    )

    target_grad_x = F.conv2d(
        target,
        sobel_x.repeat(target.shape[1], 1, 1, 1).to(target.device),
        padding=1,
        groups=target.shape[1],
    )
    target_grad_y = F.conv2d(
        target,
        sobel_y.repeat(target.shape[1], 1, 1, 1).to(target.device),
        padding=1,
        groups=target.shape[1],
    )

    pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
    target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)

    return F.mse_loss(pred_grad_mag, target_grad_mag)


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_ids=(3, 8, 15, 22), use_l1=True):
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).features.eval()
        self.layers = layer_ids
        self.use_l1 = use_l1

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def preprocess(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

    def forward(self, x, y):
        x = self.preprocess(x)
        y = self.preprocess(y)

        x_feats = self.extract_features(x)
        y_feats = self.extract_features(y)

        loss = 0
        for xf, yf in zip(x_feats, y_feats):
            if self.use_l1:
                loss += F.l1_loss(xf, yf)
            else:
                loss += F.mse_loss(xf, yf)
        return loss


class ReverseHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, input, target):
        diff = input - target
        abs_diff = torch.abs(diff)

        linear_part = abs_diff <= self.delta
        quadratic_part = ~linear_part

        loss = torch.zeros_like(abs_diff)
        loss[linear_part] = abs_diff[linear_part]
        loss[quadratic_part] = (diff[quadratic_part] ** 2 + self.delta**2) / (
            2 * self.delta
        )

        return loss.mean()


def super_resolution_task(config_name: str) -> None:
    # Configuration
    image_size = 512
    scale_factor = 4
    num_inference_steps = 6
    guidance_scale = 2.0
    lr = 5e-3
    epochs = 750
    strength = 1.0
    collect_timesteps_and_sigmas = "SD3"
    loss_type = "L1 + TV"

    # NOTE: Seed was 123
    device = torch.device("cuda")
    set_seed(123)

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
            "collect_timesteps_and_sigmas": collect_timesteps_and_sigmas,
            "loss_type": loss_type,
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

    prompt = (
        "a calico cat with green eyes in front of a white background; high resolution"
    )

    # Get prompt and pooled embeds
    prompt_embeds, pooled_prompt_embeds = sd3_pipeline.encode_prompts(
        prompt=prompt,
    )

    # Collect timesteps and sigmas
    if collect_timesteps_and_sigmas == "FMPlug":
        # Define the latent time steps here
        timesteps = sd3_pipeline.scheduler.timesteps
        sigmas = sd3_pipeline.scheduler.sigmas

        # After digging in I understand more how this works so we an set
        # a paramter to say lets start in the range of timesteps 600.0
        mask = sigmas <= strength
        timesteps = timesteps[mask]
        sigmas = sigmas[mask]

        # Now get num_inference spaced timesteps and sigmas
        num_inference_mask = torch.linspace(
            0, len(timesteps) - 1, num_inference_steps
        ).long()
        timesteps = timesteps[num_inference_mask].to(device=device, dtype=torch.float32)
        sigmas = sigmas[num_inference_mask].to(device, dtype=torch.float32)

        zero_tensor = torch.tensor([0.0], device=device, dtype=torch.float32)
        sigmas = torch.concatenate([sigmas, zero_tensor])

    elif collect_timesteps_and_sigmas == "SD3":
        timesteps, sigmas, _ = sd3_pipeline.retrieve_timesteps_and_sigmas(
            strength=strength
        )

    print(timesteps)
    print(sigmas)

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

    # Make the decoder blocks trainable
    for name, parameters in sd3_pipeline.vae.named_parameters():
        if "up_blocks.3" in name:
            parameters.requires_grad = True

    for name, parameters in sd3_pipeline.vae.named_parameters():
        if "up_blocks.2" in name:
            parameters.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {"params": [z], "lr": 1e-1},
            {"params": sd3_pipeline.vae.decoder.up_blocks[-1].parameters(), "lr": 1e-3},
            {"params": sd3_pipeline.vae.decoder.up_blocks[-2].parameters(), "lr": 1e-4},
        ]
    )

    # Export the measurment operator
    operator = img_outputs["operator"]

    # Create the loss function
    criterion = torch.nn.L1Loss()
    mse_criterion = torch.nn.MSELoss()

    # criterion = ReverseHuberLoss()
    # criterion = torch.nn.MSELoss()

    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)
    perceptual_loss_fn = VGGPerceptualLoss()
    perceptual_loss_fn = perceptual_loss_fn.to(device)

    grad_mag_fn = gradient_magnitude_loss

    # Try the grad scaler for fp16
    scaler = torch.amp.GradScaler()

    lpips_scores = []

    start = time.time()

    for idx, epoch in tqdm.tqdm(enumerate(range(epochs))):
        torch.compiler.cudagraph_mark_step_begin()

        optimizer.zero_grad()

        z0 = (z - z.mean()) / z.std()

        x_t = integrate_euler_v2(
            f=sd3_pipeline.predict,
            x0=z0,
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
        # This output is from [-1, 1]
        decoded_img = torch.clamp(
            sd3_pipeline.vae.decode(x_t, return_dict=False)[0], -1.0, 1.0
        )
        # decoded_img = decoded_img.squeeze(0).permute(1, 2, 0)

        # This puts the image in [0, 1]
        # decoded_img = (decoded_img * 0.5) + 0.5
        # decoded_img = torch.clamp(decoded_img, 0, 1)

        # Now apply the degradation
        operator_decoded_output = operator.forward(decoded_img)  # type: ignore

        # Apply the loss function - this expects [-1, 1]
        pixel_loss = criterion(operator_decoded_output, y_n)
        grad_mag_loss = grad_mag_fn(operator_decoded_output, y_n)

        mse_loss = mse_criterion(operator_decoded_output, y_n)

        data_range = 2.0
        psnr_loss = 10.0 * torch.log10((data_range**2) / mse_loss)
        # lpips_loss = lpips_loss_fn(operator_decoded_output, y_n)

        # perceptual_loss = perceptual_loss_fn(
        #     (operator_decoded_output + 1.0) / 2.0, (y_n + 1.0) / 2.0
        # )
        tv_loss = total_variation_loss(decoded_img)

        if loss_type == "L1":
            loss = pixel_loss

        elif loss_type == "L1 + TV":
            loss = pixel_loss + 0.05 * tv_loss

        elif loss_type == "L1 + Grad":
            loss = pixel_loss + 0.2 * grad_mag_loss

        elif loss_type == "PSNR":
            loss = psnr_loss

        # Update gradients of z
        scaler.scale(loss).backward()

        # Unscale gradients before clipping
        # scaler.unscale_(optimizer)

        # Clip gradients (example: max norm = 1.0)
        # torch.nn.utils.clip_grad_norm_([z], max_norm=0.05)

        scaler.step(optimizer)
        scaler.update()

        # What is the gradient norm?
        grad_norm = z.grad.norm()  # type: ignore

        # total_norm = torch.norm(decoder_model.up_blocks[-1].parameters(), 2)

        # Compute the PSNR & print loss
        with torch.no_grad():
            lpips_score = lpips_loss_fn(decoded_img, img_outputs.get("ref_img"))

            decoded_img = decoded_img.squeeze(0)
            model_img = torch.clamp((decoded_img + 1.0) / 2.0, 0.0, 1.0)  # type: ignore
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

            lpips_scores.append(lpips_score.item())

            metrics_to_log = {
                "epoch": epoch,
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

            if idx % 100 == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

                ax1.imshow(img)
                ax1.set_title("Original Image")

                ax2.imshow(model_img)
                ax2.set_title("Reconstructed Image")

                image_save_path = os.path.join(
                    save_file_path, f"reference_vs_generated_image_epoch_{idx + 1}.png"
                )
                fig.savefig(image_save_path, bbox_inches="tight")
                plt.close(fig)

    end = time.time()
    print(end - start)

    # The prompt is playing a major role in how good of a solution we can obtain
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))

    ax1.imshow(img)
    ax1.set_title("Original Image")

    model_img = exposure.equalize_adapthist(model_img, clip_limit=0.03)
    ax2.imshow(model_img)
    ax2.set_title("Reconstructed Image")

    image_diff = np.abs(img - model_img).mean(axis=-1)
    ax3.imshow(image_diff)
    ax3.set_title("Pixel Difference")

    image_save_path = os.path.join(save_file_path, "final_output.png")
    fig.savefig(image_save_path, bbox_inches="tight")

    wandb.finish()  # type: ignore
