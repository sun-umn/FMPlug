# stdlib
from typing import Callable, Dict

# third party
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# first party
from fmplug.utils.image_utils import Blurkernel, generate_tilt_map


def prepare_super_resolution_measurement(
    image_path: str,
    image_size: int,
    scale_factor: int,
    get_operator_fn: Callable,
    get_noise_fn: Callable,
    noise_sigma: float = 0.01,
    device: torch.device = torch.device("cpu"),
):
    """
    Loads and processes an image for super-resolution
    using a forward operator and noise model.

    Args:
        image_path (str): Path to the high-resolution ground truth image.
        image_size (int): Size to which the image is resized and cropped.
        scale_factor (int): Downsampling factor for the super-resolution operator.
        noise_sigma (float): Standard deviation of Gaussian noise to add.
        device (torch.device): Device to move tensors to.
        get_operator_fn (callable): Function to initialize the measurement operator.
        get_noise_fn (callable): Function to initialize the noise model.

    Returns:
        dict: A dictionary containing:
              - ref_img: torch.Tensor, original HR image in [-1, 1] range
              - y: torch.Tensor, degraded image before noise
              - y_n: torch.Tensor, degraded image after noise
              - operator: operator instance
              - noiser: noise model instance
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    img_tensor = tf(img)
    img_tensor = img_tensor * 2.0 - 1.0  # normalize to [-1, 1]
    ref_img = img_tensor.unsqueeze(0).to(device).to(torch.float32)
    ref_img.requires_grad = False

    # Operator and noise config
    config = {
        "operator": {
            "name": "super_resolution",
            "in_shape": (1, 3, image_size, image_size),
            "scale_factor": scale_factor,
        },
        "noise": {
            "name": "gaussian",
            "sigma": noise_sigma,
        },
    }

    # Initialize operator and noise model
    operator = get_operator_fn(device=device, **config["operator"])  # type
    noiser = get_noise_fn(**config["noise"])

    # Apply forward measurement model: y = A(x), y_n = A(x) + n
    y = operator.forward(ref_img)
    y_n = noiser(y)

    return {
        "ref_img": ref_img,
        "y": y,
        "y_n": y_n,
        "operator": operator,
        "noiser": noiser,
    }


def prepare_measurement(
    image_path: str,
    image_size: int,
    config: dict,
    get_operator_fn: Callable,
    get_noise_fn: Callable,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    General-purpose measurement preparation with conditional
    support for kernel and tilt.

    Args:
        image_path (str): Path to input image.
        config (dict): Dict with 'measurement', and optionally
        'kernel', 'kernel_size', 'intensity'.
        get_operator_fn (Callable): Function to initialize operator.
        get_noise_fn (Callable): Function to initialize noise.
        device (torch.device): Device for tensors.
        dtype (torch.dtype): Desired dtype.

    Returns:
        dict with: ref_img, y, y_n, operator, noiser, kernel (optional), tilt (optional)
    """
    # --- Load and normalize image ---
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    img_tensor = tf(img)
    img_tensor = img_tensor * 2.0 - 1.0  # normalize to [-1, 1]
    ref_img = img_tensor.unsqueeze(0).to(device).to(torch.float32)
    ref_img.requires_grad = False

    # --- Config unpacking ---
    measure_cfg = config["measurement"]
    op_cfg = measure_cfg["operator"]
    noise_cfg = measure_cfg["noise"]
    op_name = op_cfg.get("name", "").lower()

    # --- Optional kernel ---
    kernel = config.get("kernel", None)
    kernel_size = config.get("kernel_size", None)
    intensity = config.get("intensity", None)

    noiser = get_noise_fn(**noise_cfg)
    operator = get_operator_fn(device=device, **op_cfg)

    # --- Forward measurement model ---
    tilt = None
    if op_name == "turbulence":
        # Setup tilt
        tilt_kernel_size = 7
        tilt = generate_tilt_map(
            img_h=image_size,
            img_w=image_size,
            kernel_size=tilt_kernel_size,
            device=device,
        )
        tilt = torch.clip(tilt, -2.5, 2.5)

        # Blur kernel
        conv = Blurkernel(
            "gaussian", kernel_size=kernel_size, device=device, std=intensity
        )
        kernel = conv.get_kernel().type(torch.float32)
        kernel = kernel.to(device).view(1, 1, kernel_size, kernel_size)

        y = operator.forward(ref_img, kernel, tilt)

    elif kernel is not None:
        y = operator.forward(ref_img, kernel)

    else:
        y = operator.forward(ref_img)

    y_n = noiser(y)
    y_n.requires_grad = False

    return {  # type: ignore
        "ref_img": ref_img,
        "y": y,
        "y_n": y_n,
        "operator": operator,
        "noiser": noiser,
        "kernel": kernel,
        "kernel_size": kernel_size,
        "tilt": tilt,  # type: ignore
    }


def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=1.0, channel_axis=0)  # type: ignore
