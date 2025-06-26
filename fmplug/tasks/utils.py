# stdlib
from typing import Callable

# third party
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms


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


def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=1.0, channel_axis=0)
