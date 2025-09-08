import torch
import torch.nn as nn

class MeanVarPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        # latent net
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        # time net
        self.fc1_t = nn.Linear(1, hidden_channels)
        self.fc2_t = nn.Linear(hidden_channels, hidden_channels)
        
        # After global pooling, we add timestep
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 2)  # output: [mean, variance]

    def forward(self, x, t):
        # x: [B,C,H,W], t: [B,1] or [B]
        h = torch.relu(self.bn1(self.conv1(x)))
        h = torch.relu(self.bn2(self.conv2(h))) # [B, hidden_channels, H, W]
        h = h.mean(dim=[2, 3])  # [B, hidden_channels]
        
        # print("t: ", t.shape, "h: ", h.shape)
        
        t = torch.relu(self.fc1_t(t/1000.0))
        t = torch.relu(self.fc2_t(t))
        
        # print("t: ", t.shape, "h: ", h.shape)

        h = torch.cat([h, t], dim=1)  # [B, hidden_channels * 2]
        h = torch.relu(self.fc1(h))
        out = self.fc2(h)  # [B, 2] (mean, var)
        return out