# third party
import torch
import torch.nn as nn


class SiLU(nn.Module):
    """
    SiLU implementation - comes from the flow matching
    repository so we can reproduce their work.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
