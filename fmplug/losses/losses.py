# third party
import torch
import torch.nn as nn
import torchvision.models as models


def gram_matrix(feat):
    """
    Compute Gram matrix from a feature map.

    Args:
        feat: tensor of shape (B, C, H, W)

    Returns:
        Gram matrix of shape (B, C, C)
    """
    B, C, H, W = feat.shape
    feat = feat.view(B, C, H * W)  # (B, C, N)
    gram = torch.bmm(feat, feat.transpose(1, 2))  # (B, C, C)
    gram = gram / (C * H * W)  # normalize
    return gram


class PerceptualLossV3(nn.Module):
    def __init__(self, layers=["relu1_2", "relu2_2", "relu3_4"], layer_weights=None):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        self.vgg_layers = vgg

        self.layer_map = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_4": 17,
            "relu4_4": 26,
            "relu5_4": 35,
        }
        self.selected_layers = [self.layer_map[layer] for layer in layers]
        self.layer_weights = layer_weights or [1.0 / len(self.selected_layers)] * len(
            self.selected_layers
        )

        # Normalization constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def extract_features(self, x, selected_indices):
        feats = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in selected_indices:
                feats.append(x)
        return feats

    def forward(self, x, y):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)

        x_feats = self.extract_features(x, self.selected_layers)
        y_feats = self.extract_features(y, self.selected_layers)

        loss = 0.0
        for i in range(len(self.selected_layers)):
            loss += self.layer_weights[i] * torch.nn.functional.l1_loss(
                x_feats[i], y_feats[i]
            )

        return loss


class StyleLoss(nn.Module):
    def __init__(
        self, layers=["relu1_2", "relu2_2", "relu3_4", "relu4_4"], layer_weights=None
    ):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        self.vgg_layers = vgg

        self.layer_map = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_4": 17,
            "relu4_4": 26,
            "relu5_4": 35,
        }

        self.selected_layers = [self.layer_map[layer] for layer in layers]
        self.layer_weights = layer_weights or [1.0 / len(self.selected_layers)] * len(
            self.selected_layers
        )

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def extract_features(self, x, selected_indices):
        feats = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in selected_indices:
                feats.append(x)
        return feats

    def forward(self, input, target):
        """
        input, target: tensors of shape (B, 3, H, W) in range [0, 1]
        """
        input = (input - self.mean.to(input.device)) / self.std.to(input.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)

        input_feats = self.extract_features(input, self.selected_layers)
        target_feats = self.extract_features(target, self.selected_layers)

        loss = 0.0
        for i in range(len(self.selected_layers)):
            G_input = gram_matrix(input_feats[i])
            G_target = gram_matrix(target_feats[i])
            loss += self.layer_weights[i] * torch.nn.functional.l1_loss(
                G_input, G_target
            )

        return loss


class PerceptualLoss(nn.Module):
    def __init__(self, layers=["relu1_2", "relu2_2", "relu3_4"], layer_weights=None):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = vgg
        self.layer_map = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_4": 17,
            "relu4_4": 26,
            "relu5_4": 35,
        }
        self.selected_layers = [self.layer_map[layer] for layer in layers]
        self.layer_weights = layer_weights or [1.0 / len(self.selected_layers)] * len(
            self.selected_layers
        )

        # VGG expects normalized images
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)

        loss = 0.0
        for i, layer_idx in enumerate(self.selected_layers):
            for j in range(layer_idx + 1):
                x = self.vgg_layers[j](x)
                y = self.vgg_layers[j](y)

            loss += self.layer_weights[i] * torch.nn.functional.l1_loss(x, y)

        return loss


class FFTLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Computes L1 loss between magnitude spectra of input and target.
        Args:
            reduction (str): 'mean' or 'sum' over batch
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        input, target: tensors of shape (B, C, H, W)
        """
        # Apply 2D FFT
        input_fft = torch.fft.fft2(input, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')

        # Shift zero freq to center (optional but common)
        input_fft = torch.fft.fftshift(input_fft)
        target_fft = torch.fft.fftshift(target_fft)

        # Take magnitude spectra
        input_mag = torch.abs(input_fft)
        target_mag = torch.abs(target_fft)

        # L1 difference
        loss = torch.abs(input_mag - target_mag)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # no reduction
