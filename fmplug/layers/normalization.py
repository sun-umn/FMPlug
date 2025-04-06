# third party
import torch
import torch.nn as nn


class GroupNorm32(nn.GroupNorm):
    """
    Class to implement group normalization.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
