import torch
import torch.nn as nn


class LowRankPromptAdapter(nn.Module):
    def __init__(self, d, r=8):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(d, r))
        self.B = nn.Parameter(torch.zeros(d, r))
        nn.init.normal_(self.A, std=1e-4)
        nn.init.normal_(self.B, std=1e-4)

    def forward(self, e0):
        # e0: [batch, seq_len, d] or [batch, d]
        orig_shape = e0.shape
        e0_flat = e0.view(-1, orig_shape[-1])  # [*, d]
        delta = (e0_flat @ self.B) @ self.A.T
        return (e0_flat + delta).view(orig_shape)

