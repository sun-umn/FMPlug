# stdlib
from abc import ABC, abstractmethod
from functools import partial

# third party
import numpy as np
import yaml  # type: ignore
from torch.nn import functional as F
from torchvision import torch

# first party
from fmplug.utils.resizer import Resizer

__OPERATOR__ = {}  # type: ignore
__NOISE__ = {}  # type: ignore


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name="noise")
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device

    def forward(self, data):
        return data

    def transpose(self, data):
        return data

    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name="super_resolution")
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1 / scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)


@register_operator(name="inpainting")
class InpaintingOperator(LinearOperator):
    """This operator get pre-defined mask and return masked image."""

    def __init__(self, device):
        self.device = device

    def forward(self, data, **kwargs):
        # try:
        return data * kwargs.get("mask", None).to(self.device)
        # except:  # noqa
        #     raise ValueError("Require mask")

    def transpose(self, data, **kwargs):
        return data

    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


@register_operator(name="blind_blur")
class BlindBlurOperator(LinearOperator):
    def __init__(self, device, **kwargs) -> None:
        self.device = device

    def forward(self, data, kernel, **kwargs):
        return self.apply_kernel(data, kernel)

    def transpose(self, data, **kwargs):
        return data

    def apply_kernel(self, data, kernel):
        # TODO: faster way to apply conv?:W

        b_img = torch.zeros_like(data).to(self.device)
        for i in range(3):
            b_img[:, i, :, :] = F.conv2d(
                data[:, i : i + 1, :, :], kernel, padding="same"
            )
        return b_img


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data)


@register_operator(name="nonlinear_blur")
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)
        self.random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        # self.random_kernel.requires_grad = False

    def prepare_nonlinear_blur_model(self, opt_yml_path):
        """
        Nonlinear deblur requires external codes (bkse).
        """
        # first party
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard  # noqa

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path))
        blur_model = blur_model.to(self.device)
        for param in blur_model.parameters():
            param.requires_grad = False
        return blur_model

    def forward(self, data, **kwargs):
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=self.random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred


def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls

    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass


@register_noise(name="clean")
class Clean(Noise):
    def forward(self, data):
        return data


@register_noise(name="gaussian")
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name="poisson")
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        """
        Follow skimage.util.random_noise.
        """
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(
            np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate
        )
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)
