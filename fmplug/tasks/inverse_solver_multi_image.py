# stdlib
import glob
import os
import random
import time
from pathlib import Path

# third party
import lpips
import matplotlib.pyplot as plt
import numpy as np
import openai
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
import yaml
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.schedulers.scheduling_flow_match_heun_discrete import (
    FlowMatchHeunDiscreteScheduler,
)
from skimage.metrics import peak_signal_noise_ratio

# from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor

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


def _logit(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Safe logit for values in (0,1)
    x = x.clamp(eps, 1 - eps)
    return torch.log(x) - torch.log1p(-x)


class MonotoneTimesteps(nn.Module):
    """
    Strictly decreasing timesteps spanning [t_min, t_max].
    With learn_endpoints=True, endpoints are learned but
    constrained to 0 < t_min < t_max < 1.

    Args:
        t_init (1D Tensor | list): initial (nonincreasing) timestep vector, length >= 2.
        learn_endpoints (bool): if True, learn t_min/t_max inside (0,1)
        with t_max > t_min.
        eps (float): small numerical margin.
        check_monotone (bool): if True, assert t_init is approx. nonincreasing.
    """

    def __init__(
        self,
        t_init,
        learn_endpoints: bool = False,
        eps: float = 1e-12,
        check_monotone: bool = True,
    ):
        super().__init__()
        t_init = torch.as_tensor(t_init, dtype=torch.get_default_dtype())
        assert t_init.ndim == 1 and t_init.numel() >= 2, "t_init must be 1D with n>=2"
        n = t_init.numel()

        if check_monotone:
            if not torch.all(t_init[:-1] >= t_init[1:] - 1e-12):
                raise ValueError("t_init must be (approximately) nonincreasing.")

        # endpoints from init (we'll project into (0,1) if learnable)
        t_max_init = t_init.max()
        t_min_init = t_init.min()

        # softmax weights for positive gaps (strictly >0)
        gaps = (t_init[:-1] - t_init[1:]).clamp_min(0.0)
        w0 = (gaps + eps) / (gaps.sum() + eps * (n - 1))
        u_init = torch.log(w0)  # softmax(log w0) == w0

        self.u = nn.Parameter(u_init)  # length n-1
        self.learn_endpoints = learn_endpoints
        self.eps = eps

        if learn_endpoints:
            # --- Stick-breaking endpoints: 0 < t_min < t_max < 1 ---
            t_min_clamped = t_min_init.clamp(eps, 1 - eps)
            t_max_clamped = t_max_init.clamp(eps, 1 - eps)
            if not (t_min_clamped < t_max_clamped):
                # fallback to a valid small range if init is degenerate
                t_min_clamped = torch.tensor(0.1, dtype=t_init.dtype)
                t_max_clamped = torch.tensor(0.9, dtype=t_init.dtype)

            gap_frac = (t_max_clamped - t_min_clamped) / (1 - t_min_clamped + eps)
            gap_frac = gap_frac.clamp(eps, 1 - eps)

            # Learn logits for t_min and gap fraction
            self.a = nn.Parameter(_logit(t_min_clamped, eps))
            self.b = nn.Parameter(_logit(gap_frac, eps))
        else:
            # Fixed endpoints as buffers (clamped to [0,1])
            self.register_buffer("t_max", t_max_init.clamp(0.0, 1.0).clone())
            self.register_buffer("t_min", t_min_clamped.clamp(0.0, 1.0).clone())

    def _endpoints(self):
        """Return (t_min, t_max) with 0 < t_min < t_max < 1."""
        if self.learn_endpoints:
            eps = self.eps
            t_min = torch.sigmoid(self.a)  # (0,1)
            gap_frac = torch.sigmoid(self.b)  # (0,1)
            t_max = t_min + gap_frac * (1 - t_min)  # (t_min,1)

            # tiny safety margins; use tensor-tensor bounds for clamp
            t_min = t_min.clamp(min=eps, max=1 - 2 * eps)
            upper = torch.full_like(t_min, 1 - eps)
            t_max = torch.minimum(torch.maximum(t_max, t_min + eps), upper)
            return t_min, t_max
        else:
            return self.t_min, self.t_max

    def forward(self) -> torch.Tensor:
        # positive weights summing to 1
        w = F.softmax(self.u, dim=0)  # (n-1,)
        t_min, t_max = self._endpoints()

        # gaps that sum to (t_max - t_min)
        deltas = (t_max - t_min) * w  # (n-1,), strictly > 0

        t = torch.empty(self.u.numel() + 1, device=w.device, dtype=w.dtype)
        t[0] = t_max
        t[1:] = t_max - torch.cumsum(deltas, dim=0)

        # keep inside [0,1] (should already hold; this guards round-off)
        return t.clamp(0.001, 1.0)


class GramMatrixLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        self.device = device

        # Extract specific layers (relu1_2, relu2_2, relu3_3)
        vgg = vgg16(pretrained=True).features.to(device).eval()
        return_nodes = {
            "3": "relu1_2",  # after 2nd conv
            "8": "relu2_2",
            "15": "relu3_3",
        }
        self.feature_extractor = create_feature_extractor(vgg, return_nodes)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Register VGG mean and std buffers for normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("vgg_mean", mean)
        self.register_buffer("vgg_std", std)

    def normalize_vgg(self, x):
        return (x - self.vgg_mean) / self.vgg_std

    def forward(self, pred, target):
        # Normalize inputs before feeding into VGG
        pred = self.normalize_vgg(pred)
        target = self.normalize_vgg(target)

        pred_feats = self.feature_extractor(pred)
        target_feats = self.feature_extractor(target)

        loss = 0.0
        for key in pred_feats:
            G_pred = self.gram_matrix(pred_feats[key])
            G_target = self.gram_matrix(target_feats[key])
            loss += F.l1_loss(G_pred, G_target)
        return loss

    @staticmethod
    def gram_matrix(feat):
        B, C, H, W = feat.size()
        feat = feat.view(B, C, -1)
        G = torch.bmm(feat, feat.transpose(1, 2))  # (B, C, C)
        return G / (C * H * W)


class VGG16PerceptualLoss(nn.Module):
    def __init__(self, layers=("relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3")):
        super().__init__()
        vgg_features = vgg16(pretrained=True).features.eval()
        self.layer_name_map = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_3": 15,
            "relu4_3": 22,
            "relu5_3": 29,
        }
        self.selected_layers = layers
        self.layers_to_extract = {name: self.layer_name_map[name] for name in layers}
        self.vgg = nn.Sequential(
            *[vgg_features[i] for i in range(max(self.layers_to_extract.values()) + 1)]
        )

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        y = (y - mean) / std

        loss = 0.0
        x_feats = {}
        y_feats = {}

        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layers_to_extract.values():
                name = [k for k, v in self.layers_to_extract.items() if v == i][0]
                x_feats[name] = x
                y_feats[name] = y

        for name in self.selected_layers:
            loss += nn.functional.l1_loss(x_feats[name], y_feats[name])
        return loss


class MultiChannelLoss(torch.nn.Module):
    def forward(self, y_pred, y_true):
        # These will be multi-channeled
        y_pred = y_pred.squeeze(0)
        y_true = y_true.squeeze(0)

        first_channel_loss = torch.abs(y_pred[0, :, :] - y_true[0, :, :]).mean()
        second_channel_loss = torch.abs(y_pred[1, :, :] - y_true[1, :, :]).mean()
        third_channel_loss = torch.abs(y_pred[2, :, :] - y_true[2, :, :]).mean()

        return first_channel_loss + second_channel_loss + third_channel_loss


class ChannelExpandCompress(nn.Module):
    def __init__(self, in_channels=16, mid_channels=256, out_channels=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels, affine=True),
        )

    def forward(self, z):
        return self.conv(z)


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


# Compute cosine similarity per layer
def cosine_similarity_per_layer(feats0, feats1, eps=1e-8):
    cos_sims = []
    for f0, f1 in zip(feats0, feats1):
        B, C, H, W = f0.shape
        f0_flat = f0.view(B, C, -1)
        f1_flat = f1.view(B, C, -1)
        f0_norm = F.normalize(f0_flat, dim=1, eps=eps)
        f1_norm = F.normalize(f1_flat, dim=1, eps=eps)
        cos_sim = (f0_norm * f1_norm).sum(dim=1).mean(dim=1)  # [B]
        cos_sims.append(cos_sim)
    return torch.stack(cos_sims, dim=1).squeeze(0)  # shape: [num_layers]


# Compute LPIPS loss per layer (unnormalized, for comparison)
def lpips_loss_per_layer(feats0, feats1, net):
    losses = []
    for f0, f1, weight in zip(feats0, feats1, net.lins):
        diff = (f0 - f1) ** 2
        weighted = weight.model[1].weight.view(1, -1, 1, 1) * diff
        loss = weighted.sum(dim=1).mean()
        losses.append(loss.item())
    return losses


def generate_prompt_with_openai_vlm(image_path, api_key):
    """
    Generate a 77-token prompt using OpenAI's VLM (GPT-4V) for the given image.

    Args:
        image_path (str): Path to the image file
        api_key (str): OpenAI API key

    Returns:
        str: Generated prompt
    """
    client = openai.OpenAI(api_key=api_key)
    text = (
        (
            "Describe this image in detail. Focus on visual elements, "
            "composition, colors, lighting, and any notable features. "
            "Keep the description concise but comprehensive, suitable for "
            "generating a similar image with an AI model. Limit to "
            "approximately 77 tokens."
        ),
    )

    try:
        with open(image_path, "rb") as image_file:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_file.read()}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=150,
            )

        prompt = response.choices[0].message.content.strip()
        return prompt

    except Exception as e:
        print(f"Error generating prompt for {image_path}: {e}")
        # Fallback prompt
        return (
            "A high-quality photograph with natural lighting and detailed composition"
        )


def plot_sigma_trajectories(initial_sigmas, final_sigmas, save_path, image_name):
    """
    Plot the initial vs final sigma trajectories to show how they changed over time.

    Args:
        initial_sigmas (torch.Tensor): Initial sigma values
        final_sigmas (torch.Tensor): Final sigma values after training
        save_path (str): Directory to save the plot
        image_name (str): Name of the image for the plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Convert to numpy for plotting
    initial_sigmas_np = initial_sigmas.detach().cpu().numpy()
    final_sigmas_np = final_sigmas.detach().cpu().numpy()

    # Create x-axis (timestep indices)
    x = np.arange(len(initial_sigmas_np))

    # Plot both trajectories
    ax.plot(x, initial_sigmas_np, "b-", label="Initial Sigmas", linewidth=2)
    ax.plot(x, final_sigmas_np, "r-", label="Final Sigmas", linewidth=2)

    ax.set_xlabel("Timestep Index")
    ax.set_ylabel("Sigma Value")
    ax.set_title(f"Sigma Trajectory Evolution - {image_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save the plot
    plot_path = os.path.join(save_path, f"sigma_trajectory_{image_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Sigma trajectory plot saved to: {plot_path}")


def super_resolution_multi_image_task(config_name: str) -> None:
    # Load configuration from YAML file
    base_config_path = "fmplug/configs/super_resolution"
    config_file_path = os.path.join(base_config_path, config_name + ".yaml")

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    # Extract configuration parameters
    image_size = config["image_size"]
    scale_factor = config["scale_factor"]
    num_inference_steps = config["num_inference_steps"]
    guidance_scale = config["guidance_scale"]
    lr = config["lr"]
    epochs = config["epochs"]
    strength = config["strength"]
    ode_solver = config["ode_solver"]
    shift = config.get("shift", 3.0)
    device = torch.device(config["device"])
    seed = config["seed"]
    noise_sigma = config["noise_sigma"]
    image_directory = config[
        "image_directory"
    ]  # New: directory containing multiple images
    wandb_project = config["wandb_project"]
    wandb_tags = config["wandb_tags"]
    save_base_path = config["save_base_path"]
    openai_api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))

    # Check if custom prompt is provided in config
    custom_prompt = config.get("prompt", "")

    # Set random seed
    set_seed(seed)

    # Global variables for wandb
    API_KEY = os.environ.get("WANDB_API_KEY")

    # Enable wandb
    print("Initialize Project ...")
    wandb.login(key=API_KEY)  # type: ignore

    wandb_instance = wandb.init(  # type: ignore
        # set the wandb project where this run will be logged
        project=wandb_project,
        tags=wandb_tags,
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
            "image_directory": image_directory,
            "use_mp": False,
        },
    )

    # Create the directory to save all of the model results
    wandb_experiment_id = wandb_instance.id
    save_file_path = os.path.join(save_base_path, wandb_experiment_id)
    os.makedirs(save_file_path, exist_ok=True)

    # Create prompts directory
    prompts_dir = os.path.join(save_file_path, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)

    # Initialize metrics collection for CSV export
    metrics_data = []

    # Get all PNG images from the directory
    image_pattern = os.path.join(image_directory, "*.png")
    image_paths = glob.glob(image_pattern)

    if not image_paths:
        raise ValueError(f"No PNG images found in directory: {image_directory}")

    print(f"Found {len(image_paths)} images to process")

    print("Load in SD3 image to image model ...")
    if ode_solver == "heun":
        sd3_pipeline = StableDiffusion3BaseV2(
            model_key="stabilityai/stable-diffusion-3-medium-diffusers",
            scheduler=FlowMatchHeunDiscreteScheduler(),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )
        # gradient_clipping_value = 0.005
        print(sd3_pipeline.scheduler.config)

    elif ode_solver == "euler":
        sd3_pipeline = StableDiffusion3BaseV2(
            model_key="stabilityai/stable-diffusion-3-medium-diffusers",
            scheduler=FlowMatchEulerDiscreteScheduler(
                shift=shift
                # use_dynamic_shifting=True,
                # base_shift=10.0,
                # max_shift=0.80,
                # use_karras_sigmas=True,
            ),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
        )
        # gradient_clipping_value = 0.006

    # Create the loss function
    criterion = torch.nn.L1Loss()
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)

    # Process each image
    for image_idx, image_path in enumerate(image_paths):
        print(f"\nProcessing image {image_idx + 1}/{len(image_paths)}: {image_path}")

        # Extract image name for saving
        image_name = Path(image_path).stem

        # Generate or use custom prompt
        if custom_prompt:
            # Use custom prompt from config
            prompt = custom_prompt
            print(f"Using custom prompt for {image_name}: {prompt}")
        else:
            # Generate prompt using OpenAI VLM
            print(f"Generating prompt for {image_name}...")
            prompt = generate_prompt_with_openai_vlm(image_path, openai_api_key)
            print(f"Generated prompt: {prompt}")

        # Save prompt to text file
        prompt_file_path = os.path.join(prompts_dir, f"{image_name}_prompt.txt")
        with open(prompt_file_path, "w") as f:
            f.write(f"{image_name}, {prompt}")
        print(f"Prompt saved to: {prompt_file_path}")

        # Load an image and simulate the measurement process
        img_outputs = prepare_super_resolution_measurement(
            image_path=image_path,
            image_size=image_size,
            scale_factor=scale_factor,
            noise_sigma=noise_sigma,
            device=device,
            get_operator_fn=get_operator,
            get_noise_fn=get_noise,
        )

        # Get prompt and pooled embeds
        prompt_embeds, pooled_prompt_embeds = sd3_pipeline.encode_prompts(
            prompt=prompt,
        )

        # Collect timesteps and sigmas
        print(sd3_pipeline.scheduler.sigmas)
        timesteps, sigmas, _ = sd3_pipeline.retrieve_timesteps_and_sigmas(
            strength=strength, height=image_size, width=image_size
        )

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
            latents = (
                latents / latents.norm() * torch.sqrt(torch.tensor(latents.numel()))
            )
            z = latents

        # Make z trainable
        z = z.requires_grad_(True)

        # We will log this with wandb
        initial_z_min = z.min()
        initial_z_max = z.max()
        initial_z_mean = z.mean()

        # We can initialize the new timesteps and sigmas here
        print(sigmas, len(sigmas))
        print(timesteps, len(timesteps))
        # stdlib
        mono = MonotoneTimesteps(sigmas, learn_endpoints=True).to(device)
        mono.train()

        # Store initial sigmas for comparison
        initial_sigmas = mono().clone().detach()

        optimizer = torch.optim.Adam(
            [
                {"params": [z], "lr": lr},
                {"params": mono.parameters(), "lr": 1e-2},
            ]
        )

        # Export the measurment operator
        operator = img_outputs["operator"]

        start = time.time()
        for idx, epoch in tqdm.tqdm(enumerate(range(epochs))):
            # # stdlib
            sigmas = mono()
            timesteps = sigmas * 1000.0

            # Filter timesteps here
            init_timestep = min(num_inference_steps * strength, num_inference_steps)
            t_start = int(max(num_inference_steps - init_timestep, 0))
            timesteps = timesteps[t_start * sd3_pipeline.scheduler.order : -1]

            print(sigmas, len(sigmas))
            print(timesteps, len(timesteps))
            # torch.compiler.cudagraph_mark_step_begin()

            optimizer.zero_grad()

            # The transformer at least expects a zero mean
            # as an input but can handle standard and non-standard
            # gaussian distributions
            z0 = (z - z.mean()) / z.std()
            # z0 = model(z)

            # with torch.amp.autocast("cuda", dtype=torch.float16):
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
            loss = criterion(
                operator_decoded_output, y_n
            )  # + criterion(enc_dec_op, enc_y_n)

            # Update gradients of z
            # scaler.scale(loss).backward()
            loss.backward()

            # Unscale gradients before clipping
            # scaler.unscale_(optimizer)

            # Clip gradients (example: max norm = 1.0)
            # Found this value to work well for Heun2 - may need to be tuned
            # for euler
            # if idx <= 10:
            #     torch.nn.utils.clip_grad_norm_([z], max_norm=0.01)

            # else:
            torch.nn.utils.clip_grad_norm_([z], max_norm=0.005)

            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()

            # What is the gradient norm?
            # Compute the grad norm so we can track it
            grad_norm = z.grad.norm()  # type: ignore

            # Compute the PSNR & print loss
            with torch.no_grad():
                lpips_score = lpips_loss_fn(decoded_img, img_outputs.get("ref_img"))

                # Get the features
                feats0 = lpips_loss_fn.net.forward(img_outputs.get("ref_img"))
                feats1 = lpips_loss_fn.net.forward(decoded_img.to(dtype=torch.float32))

                decoded_img = decoded_img.squeeze(0)
                model_img = (decoded_img + 1.0) / 2.0  # type: ignore
                model_img = model_img.squeeze(0).detach().cpu().numpy().astype("float")  # type: ignore

                img = img_outputs.get("ref_img")
                img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)  # type: ignore
                img = img.squeeze(0).detach().cpu().numpy().astype("float")  # type: ignore

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
                    "image_name": image_name,
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
                if (idx <= 25) or (idx % 100 == 0):
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
                        save_file_path,
                        f"{image_name}_reference_vs_generated_image_epoch_{idx + 1}.png",
                    )

                    fig.tight_layout()
                    fig.savefig(image_save_path, bbox_inches="tight")
                    plt.close(fig)

        end = time.time()
        print(f"Training time for {image_name}: {end - start}")

        # Get final sigmas for trajectory comparison
        final_sigmas = mono().clone().detach()

        # Plot sigma trajectories
        plot_sigma_trajectories(
            initial_sigmas, final_sigmas, save_file_path, image_name
        )

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

        image_save_path = os.path.join(save_file_path, f"{image_name}_final_output.png")
        fig.tight_layout()
        fig.savefig(image_save_path, bbox_inches="tight")

        # Get the plots for lpips loss
        cos_sim_values = cosine_similarity_per_layer(feats0, feats1)
        lpips_losses = lpips_loss_per_layer(feats0, feats1, lpips_loss_fn)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Cosine similarity plot
        ax1.plot(
            np.arange(len(cos_sim_values)), cos_sim_values.cpu().numpy(), marker="o"
        )
        ax1.set_title("Cosine Similarity Per LPIPS Layer")
        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Cosine Similarity")

        ax2.plot(np.arange(len(lpips_losses)), lpips_losses, marker="x", color="r")
        ax2.set_title("LPIPS Contribution Per Layer")
        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Layerwise LPIPS Loss")

        image_save_path = os.path.join(
            save_file_path, f"{image_name}_lpips_assessment.png"
        )
        fig.tight_layout()
        fig.savefig(image_save_path, bbox_inches="tight")

        # Store final metrics for CSV export
        final_metrics = {
            "image_name": image_name,
            "mse": mse_score,
            "psnr": psnr_score,
            "ssim": ssim_score,
            "lpips": lpips_score.item(),
        }
        metrics_data.append(final_metrics)

    # Save all metrics to CSV using polars
    if metrics_data:
        df = pl.DataFrame(metrics_data)
        csv_path = os.path.join(save_file_path, "final_metrics.csv")
        df.write_csv(csv_path)
        print(f"Final metrics saved to: {csv_path}")

        # Print summary statistics
        print("\nFinal Metrics Summary:")
        print(f"Number of images processed: {len(metrics_data)}")
        print(f"Average PSNR: {df['psnr'].mean():.2f}")
        print(f"Average SSIM: {df['ssim'].mean():.4f}")
        print(f"Average LPIPS: {df['lpips'].mean():.4f}")
        print(f"Average MSE: {df['mse'].mean():.6f}")

    wandb.finish()  # type: ignore
