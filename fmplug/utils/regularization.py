import torch
from fmplug.models.norm_net import MeanVarPredictor
import math

def gauss_sphere_reg(z, low=0.975, high=1.025):
    with torch.no_grad():
        norm = torch.norm(z, p=2)
        target = math.sqrt(z.numel())

        if norm < low * target:
            z.data = z / norm * target * low
        elif norm > high * target:
            z.data = z / norm * target * high
        else:
            z.data = z
    return z
