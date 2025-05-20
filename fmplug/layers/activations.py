# stdlib
import math

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


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Uniform initialization for first layer
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # SIREN-specific init
                self.linear.weight.uniform_(
                    -math.sqrt(6 / self.in_features) / self.omega_0,
                    math.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class CoordFeatureSiren(nn.Module):
    def __init__(
        self, feature_dim=16, hidden_dim=64, out_dim=16, depth=3, omega_0=30.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(feature_dim + 2, hidden_dim, is_first=True, omega_0=omega_0),
            *[
                SineLayer(hidden_dim, hidden_dim, omega_0=omega_0)
                for _ in range(depth - 2)
            ],
            # Removed from suggestion because we want the final output as -1 to 1
            # nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # Coord grid
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
        coords = coords.unsqueeze(0).expand(B, H, W, 2)  # (B, H, W, 2)

        # Flatten and concat
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        coords_flat = coords.reshape(B * H * W, 2)
        input_flat = torch.cat([x_flat, coords_flat], dim=-1)

        # Apply SIREN
        out_flat = self.net(input_flat)
        out = out_flat.view(B, H, W, -1).permute(0, 3, 1, 2)  # Back to (B, C, H, W)
        return out
