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
from skimage import color, exposure
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


def rgb_to_lab_torch(rgb_tensor):
    """
    Convert RGB tensor to LAB color space
    rgb_tensor: (B, 3, H, W) in range [-1, 1]
    """
    # Convert to [0, 1] range for skimage
    rgb_01 = (rgb_tensor + 1) / 2

    batch_size, channels, height, width = rgb_tensor.shape
    lab_batch = torch.zeros_like(rgb_tensor)

    for b in range(batch_size):
        # Convert to numpy (H, W, C) format
        rgb_np = rgb_01[b].permute(1, 2, 0).detach().cpu().numpy()

        # Convert to LAB
        lab_np = color.rgb2lab(rgb_np)

        # Convert back to tensor (C, H, W)
        lab_tensor = torch.from_numpy(lab_np).permute(2, 0, 1).to(rgb_tensor.device)
        lab_batch[b] = lab_tensor

    return lab_batch


def lab_to_rgb_torch(lab_tensor):
    """
    Convert LAB tensor back to RGB
    """
    batch_size, channels, height, width = lab_tensor.shape
    rgb_batch = torch.zeros_like(lab_tensor)

    for b in range(batch_size):
        # Convert to numpy (H, W, C) format
        lab_np = lab_tensor[b].permute(1, 2, 0).detach().cpu().numpy()

        # Convert to RGB
        rgb_np = color.lab2rgb(lab_np)

        # Convert back to tensor (C, H, W)
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).to(lab_tensor.device)
        rgb_batch[b] = rgb_tensor

    # Convert back to [-1, 1] range
    return rgb_batch * 2 - 1


def luminance_histogram_equalization(rgb_tensor, method="adaptive"):
    """
    Apply histogram equalization only to luminance, preserving colors

    Args:
        rgb_tensor: (B, 3, H, W) in range [-1, 1]
        method: 'adaptive' (CLAHE) or 'global'

    Returns:
        enhanced_tensor: (B, 3, H, W) in range [-1, 1]
    """
    # Convert to LAB color space
    lab_tensor = rgb_to_lab_torch(rgb_tensor)

    batch_size, channels, height, width = lab_tensor.shape
    enhanced_lab = lab_tensor.clone()

    for b in range(batch_size):
        # Extract L channel (luminance)
        l_channel = lab_tensor[b, 0].detach().cpu().numpy()  # L channel is [0, 100]

        # Normalize L channel to [0, 1] for skimage
        l_normalized = l_channel / 100.0

        # Apply histogram equalization to luminance only
        if method == "adaptive":
            l_equalized = exposure.equalize_adapthist(
                l_normalized,
                kernel_size=64,  # Local region size
                clip_limit=0.01,  # Prevents over-amplification
                nbins=256,
            )
        else:
            l_equalized = exposure.equalize_hist(l_normalized, nbins=256)

        # Convert back to [0, 100] range
        l_equalized = l_equalized * 100.0

        # Put enhanced L channel back
        enhanced_lab[b, 0] = torch.from_numpy(l_equalized).to(rgb_tensor.device)

    # Convert back to RGB
    enhanced_rgb = lab_to_rgb_torch(enhanced_lab)

    return enhanced_rgb


def hsv_value_enhancement(rgb_tensor, enhancement_factor=1.3):
    """
    Enhance only the Value component in HSV space
    """
    # Convert to [0, 1] range
    rgb_01 = (rgb_tensor + 1) / 2

    batch_size, channels, height, width = rgb_tensor.shape
    enhanced_batch = torch.zeros_like(rgb_tensor)

    for b in range(batch_size):
        # Convert to numpy (H, W, C)
        rgb_np = rgb_01[b].permute(1, 2, 0).detach().cpu().numpy()

        # Convert to HSV
        hsv_np = color.rgb2hsv(rgb_np)

        # Enhance Value channel with gamma correction
        hsv_np[:, :, 2] = np.power(hsv_np[:, :, 2], 1.0 / enhancement_factor)
        hsv_np[:, :, 2] = np.clip(hsv_np[:, :, 2], 0, 1)

        # Convert back to RGB
        rgb_enhanced = color.hsv2rgb(hsv_np)

        # Convert back to tensor
        enhanced_batch[b] = (
            torch.from_numpy(rgb_enhanced).permute(2, 0, 1).to(rgb_tensor.device)
        )

    # Convert back to [-1, 1] range
    return enhanced_batch * 2 - 1


def adaptive_contrast_enhancement(rgb_tensor, method="lab_clahe"):
    """
    Multiple color-preserving contrast enhancement methods

    Args:
        method: 'lab_clahe', 'hsv_enhance', 'gamma_correction', 'unsharp_mask'
    """
    if method == "lab_clahe":
        return luminance_histogram_equalization(rgb_tensor, method="adaptive")

    elif method == "hsv_enhance":
        return hsv_value_enhancement(rgb_tensor, enhancement_factor=1.3)

    elif method == "gamma_correction":
        # Simple gamma correction
        rgb_01 = (rgb_tensor + 1) / 2
        gamma = 0.8  # < 1 brightens, > 1 darkens
        enhanced = torch.pow(rgb_01, gamma)
        return enhanced * 2 - 1

    elif method == "unsharp_mask":
        return apply_unsharp_mask(rgb_tensor, amount=1.5, radius=1.0)

    else:
        raise ValueError(f"Unknown method: {method}")


def apply_unsharp_mask(image, amount=1.5, radius=1.0, threshold=0.01):
    """
    Apply unsharp masking for contrast enhancement
    """
    # Create Gaussian kernel for blurring
    kernel_size = int(radius * 4 + 1)
    sigma = radius

    # Generate 1D Gaussian
    coords = torch.arange(kernel_size, dtype=torch.float32, device=image.device)
    coords = coords - (kernel_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    # Create 2D kernel
    kernel = g.outer(g).view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(image.shape[1], 1, 1, 1)

    # Apply Gaussian blur
    blurred = F.conv2d(image, kernel, padding=kernel_size // 2, groups=image.shape[1])

    # Create detail layer
    detail = image - blurred

    # Apply threshold to avoid enhancing noise
    detail_mask = (torch.abs(detail) > threshold).float()
    detail = detail * detail_mask

    # Add enhanced details back
    enhanced = image + amount * detail

    # Clamp to valid range
    return torch.clamp(enhanced, -1, 1)


def compare_enhancement_methods(rgb_tensor):
    """
    Compare different enhancement methods
    """
    methods = {
        "original": rgb_tensor,
        "lab_clahe": adaptive_contrast_enhancement(rgb_tensor, "lab_clahe"),
        "hsv_enhance": adaptive_contrast_enhancement(rgb_tensor, "hsv_enhance"),
        "gamma_correction": adaptive_contrast_enhancement(
            rgb_tensor, "gamma_correction"
        ),
        "unsharp_mask": adaptive_contrast_enhancement(rgb_tensor, "unsharp_mask"),
    }

    return methods


def high_frequency_loss(current_img, reference_img):
    """
    Compute high-frequency loss between two images using Laplacian filter.

    This loss helps preserve fine details and edges by comparing the high-frequency
    content of the current image with a reference image.

    Args:
        current_img: torch.Tensor of shape (B, C, H, W)
                    Current image being optimized
        reference_img: torch.Tensor of shape (B, C, H, W)
                      Reference image (e.g., from SD3)
                      Must have same shape as current_img

    Returns:
        torch.Tensor: Scalar loss value

    Input Range: Any range works ([-1,1], [0,1], [0,255]) but both inputs
                should be in the same range

    Example:
        >>> current = torch.randn(1, 3, 512, 512) * 2 - 1  # [-1, 1]
        >>> reference = torch.randn(1, 3, 512, 512) * 2 - 1
        >>> loss = high_frequency_loss(current, reference)
        >>> print(loss.shape)  # torch.Size([])
    """

    # Validate inputs
    assert (
        current_img.shape == reference_img.shape
    ), f"Shape mismatch: {current_img.shape} vs {reference_img.shape}"

    assert (
        len(current_img.shape) == 4
    ), f"Expected 4D tensor (B,C,H,W), got {len(current_img.shape)}D"

    # Get device and shape info
    device = current_img.device
    batch_size, num_channels, height, width = current_img.shape

    # Create Laplacian kernel for edge detection
    # This kernel detects edges and high-frequency content
    laplacian_kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    # Repeat kernel for each channel (groups=num_channels for per-channel convolution)
    laplacian_kernel = laplacian_kernel.repeat(num_channels, 1, 1, 1)

    # Apply Laplacian filter to extract high-frequency content
    current_hf = F.conv2d(
        current_img,
        laplacian_kernel,
        padding=1,
        groups=num_channels,  # Apply same kernel to each channel independently
    )

    reference_hf = F.conv2d(
        reference_img, laplacian_kernel, padding=1, groups=num_channels
    )

    # Compute MSE loss between high-frequency components
    hf_loss = F.mse_loss(current_hf, reference_hf)

    return hf_loss


def super_resolution_task(config_name: str) -> None:
    # Configuration
    image_size = 512
    scale_factor = 4
    num_inference_steps = 7
    guidance_scale = 2.0
    lr = 1e-2
    epochs = 5000
    strength = 1.0
    collect_timesteps_and_sigmas = "FMPlug"
    loss_type = "L1 + TV"

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
        image_path="/users/5/dever120/FMPlug/data/ffhq_baby.png",
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

    # prompt = (
    #     "a calico cat with green eyes in front of a white background; high resolution"
    # )
    # prompt = (
    #     "A striking close-up portrait of a calico cat with vivid, "
    #     "symmetrical facial markings. The cat's fur is a mix of deep black, "
    #     "snowy white, and vibrant orange patches. Its alert ears are black "
    #     "with subtle white tufts on the inner side, while a bold orange blaze "
    #     "runs vertically between its eyes. The eyes are large, round, and "
    #     "expressive, glowing a piercing chartreuse green with sharp black "
    #     "slit pupils, reflecting ambient light. The cat's pink nose is "
    #     "centered on its white muzzle, which contrasts sharply with the "
    #     "surrounding dark fur. There is a small orange spot on the left cheek "
    #     "and a distinctive black spot under the chin, enhancing the cat’s "
    #     "unique appearance. The cat stares directly into the camera with an "
    #     "intense, inquisitive gaze. The background is clean and white, making "
    #     "the cat the clear focal point of the image. The lighting is even and "
    #     "soft, highlighting the texture of the cat’s fur and the depth in its "
    #     "eyes. The composition is tight and symmetrical, emphasizing the cat's "
    #     "face in a high-resolution, naturalistic style. No accessories, no "
    #     "distractions—just a calm, elegant feline gaze."
    # )

    prompt = (
        "A close-up portrait of an adorable baby lying on a plush blanket, "
        "soft natural light gently streaming across their face. The baby has "
        "smooth, glowing skin, round cheeks, and expressive deep eyes that "
        "gaze curiously forward. Their short black hair is neatly combed, "
        "slightly tousled near the crown. They wear a soft green and white "
        "striped onesie with a comfortable, relaxed fit. The background is a "
        "cozy, vibrant aqua blue cushion that adds warmth and serenity to "
        "the scene. The baby appears calm and alert, with a subtle hint of "
        "a smile forming on their lips. The lighting emphasizes the soft "
        "features of their face, creating gentle highlights and shadows. "
        "Photographed with a shallow depth of field, the background is "
        "pleasantly blurred, drawing focus to the baby’s delicate features. "
        "There’s a soothing atmosphere, evoking feelings of peace, innocence, "
        "and tenderness. The composition is balanced, centering the baby’s "
        "face with careful attention to natural textures and colors. A faint "
        "blue beam of light diagonally crosses the image, adding a dreamy, "
        "ethereal quality. The image should reflect ultra-realistic detail, "
        "with lifelike textures in the skin, fabric, and lighting. Emphasize "
        "the softness of the environment and the purity of the baby’s "
        "expression. Render in high resolution, cinematic color grading, "
        "diffused light, 85mm lens, f/1.4, shallow depth of field, ultra-"
        "realistic detail, natural skin tones, soft bokeh, hyperrealist style."
    )

    # prompt = (
    #     "Close-up of a calico cat with green eyes, vivid fur markings, "
    #     "white background, soft lighting, centered gaze, high detail."
    # )

    # Get prompt and pooled embeds
    prompt_embeds, pooled_prompt_embeds = sd3_pipeline.encode_prompts(
        prompt=prompt,
    )

    # Collect timesteps and sigmas
    print(sd3_pipeline.scheduler.sigmas)
    if collect_timesteps_and_sigmas == "FMPlug":
        # Define the latent time steps here
        timesteps = sd3_pipeline.scheduler.timesteps
        sigmas = sd3_pipeline.scheduler.sigmas
        print(sigmas)

        # After digging in I understand more how this works so we an set
        # a paramter to say lets start in the range of timesteps 600.0
        min_strength = 0.10
        mask = (sigmas <= strength) & (sigmas >= min_strength)
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
        # z = sd3_pipeline.prepare_latents(latents=None)

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

    # prompt_embeds = prompt_embeds.requires_grad_(True)
    # pooled_prompt_embeds.requires_grad_(True)

    # Make the decoder blocks trainable
    for name, parameters in sd3_pipeline.vae.named_parameters():
        if "up_blocks.3" in name:
            parameters.requires_grad = True

    for name, parameters in sd3_pipeline.vae.named_parameters():
        if "up_blocks.2" in name:
            parameters.requires_grad = True

    # for name, parameters in sd3_pipeline.vae.named_parameters():
    #     if "up_blocks.1" in name:
    #         parameters.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {"params": [z], "lr": lr},
            {"params": sd3_pipeline.vae.decoder.up_blocks[-1].parameters(), "lr": 1e-3},
            {"params": sd3_pipeline.vae.decoder.up_blocks[-2].parameters(), "lr": 1e-4},
        ]
    )

    # optimizer = torch.optim.Adam(
    #     [
    #         {"params": [z], "lr": lr},
    #         # {"params": prompt_embeds, "lr": 1e-1},
    #         # {"params": pooled_prompt_embeds, "lr": 1e-1},
    #     ]
    # )

    # optimizer = torch.optim.Adam(
    #     [
    #         {"params": [z], "lr": lr},
    #     ]
    # )

    # Export the measurment operator
    operator = img_outputs["operator"]

    # Create the loss function
    criterion = torch.nn.L1Loss()
    # mse_criterion = torch.nn.MSELoss()
    # huber_criterion = nn.HuberLoss(delta=0.5)

    # criterion = ReverseHuberLoss()
    # criterion = torch.nn.MSELoss()

    lpips_loss_fn = lpips.LPIPS(net="vgg").to(device)
    perceptual_loss_fn = VGGPerceptualLoss()
    perceptual_loss_fn = perceptual_loss_fn.to(device)

    # grad_mag_fn = gradient_magnitude_loss

    # Try the grad scaler for fp16
    scaler = torch.amp.GradScaler()

    lpips_scores = []

    start = time.time()

    for idx, epoch in tqdm.tqdm(enumerate(range(epochs))):
        torch.compiler.cudagraph_mark_step_begin()

        optimizer.zero_grad()

        # z0 = (z - z.mean()) / z.std()
        # z0 = z - z.mean()

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
        # channel_loss = (
        #     (operator_decoded_output.squeeze(0) - y_n.squeeze(0) ** 2)
        #     .sum(axis=0)
        #     .mean()
        # )
        # hf_loss = high_frequency_loss(operator_decoded_output, y_n)

        # grad_mag_loss = grad_mag_fn(operator_decoded_output, y_n)

        # mse_loss = mse_criterion(operator_decoded_output, y_n)

        # data_range = 2.0
        # psnr_loss = 10.0 * torch.log10((data_range**2) / mse_loss)
        # lpips_loss = lpips_loss_fn(operator_decoded_output, y_n)

        # perceptual_loss = perceptual_loss_fn(
        #     (operator_decoded_output + 1.0) / 2.0, (y_n + 1.0) / 2.0
        # )
        tv_loss = total_variation_loss(decoded_img)

        if loss_type == "L1":
            loss = pixel_loss

        elif loss_type == "L1 + TV":
            print(
                pixel_loss.item(),
                # channel_loss.item(),
            )
            loss = pixel_loss + 0.1 * tv_loss  # + 0.05 * lpips_loss

        # elif loss_type == "L1 + Grad":
        #     loss = pixel_loss + 0.2 * grad_mag_loss

        # elif loss_type == "PSNR":
        #     loss = psnr_loss

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
        # prompt_embed_grad = prompt_embeds.grad.norm()
        # pooled_prompt_embed_grad = pooled_prompt_embeds.grad.norm()

        # total_norm = torch.norm(decoder_model.up_blocks[-1].parameters(), 2)

        # Compute the PSNR & print loss
        with torch.no_grad():
            lpips_score = lpips_loss_fn(decoded_img, img_outputs.get("ref_img"))
            eq_model_img = adaptive_contrast_enhancement(
                decoded_img, method="lab_clahe"
            )

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

            lpips_scores.append(lpips_score.item())

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
                # "prompt_embed_grad": prompt_embed_grad,
                # "pooled_prompt_embed_grad": pooled_prompt_embed_grad,
            }
            wandb.log(metrics_to_log)  # type: ignore

            eq_model_img = (
                eq_model_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            )
            eq_model_img = (eq_model_img + 1.0) / 2.0

            if idx % 50 == 0:
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 10))

                ax1.imshow(img)
                ax1.set_title("Original Image")

                ax2.imshow(model_img)
                ax2.set_title("Reconstructed Image")

                ax3.imshow(eq_model_img)
                ax3.set_title("Rec. Img + Hist. Eq")

                image_diff = ((img - model_img) ** 2).mean(axis=-1)
                print(
                    image_diff.min(),
                    image_diff.max(),
                    image_diff.mean(),
                )

                ax4.imshow(image_diff)
                ax4.set_title("Pixel Difference")

                image_save_path = os.path.join(
                    save_file_path, f"reference_vs_generated_image_epoch_{idx + 1}.png"
                )
                fig.savefig(image_save_path, bbox_inches="tight")
                plt.close(fig)

    end = time.time()
    print(end - start)

    # The prompt is playing a major role in how good of a solution we can obtain
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 15))

    ax1.imshow(img)
    ax1.set_title("Original Image")

    ax2.imshow(model_img)
    ax2.set_title("Reconstructed Image")

    ax3.imshow(eq_model_img)
    ax3.set_title("Rec. Img + Hist. Eq")

    image_diff = np.abs(img - model_img).mean(axis=-1)
    ax4.imshow(image_diff)
    ax4.set_title("Pixel Difference")

    image_save_path = os.path.join(save_file_path, "final_output.png")
    fig.tight_layout()
    fig.savefig(image_save_path, bbox_inches="tight")

    wandb.finish()  # type: ignore
