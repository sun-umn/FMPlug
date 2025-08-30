# third party
import torch
import torch.nn as nn
import torch.nn.functional as F


class STEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # return torch.sign(torch.clamp(input, min=-1.0, max=1.0))
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Bypass the non differterable operations
        # return grad_output
        return F.hardtanh(grad_output)


class STE(nn.Module):
    def forward(self, input):
        return STEFunc.apply(input)


class Block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, pad='same'):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size, padding=pad)
        self.batchnorm1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size, padding=pad)
        self.batchnorm2 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        # with torch.autograd.graph.save_on_cpu(pin_memory=True):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channels=[3, 64, 128, 256, 512, 1024], kernel_size=3):
        super().__init__()
        self.encode_blocks = nn.ModuleList(
            [
                Block(channels[i], channels[i + 1], kernel_size)
                for i in range(len(channels) - 1)
            ]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.encode_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class Decoder(nn.Module):
    def __init__(self, channels=[1024, 512, 256, 128, 64], skip=[]):
        super().__init__()
        self.channels = channels
        self.skip = skip

        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                for i in range(len(channels) - 1)
            ]  # kernel_size used to be 4
        )
        self.decode_blocks = []
        for i in range(len(channels) - 1):
            if i in self.skip:
                self.decode_blocks.append(Block(2 * channels[i + 1], channels[i + 1]))
            else:
                self.decode_blocks.append(Block(channels[i + 1], channels[i + 1]))
        self.decode_blocks = nn.ModuleList(self.decode_blocks)

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            if i in self.skip:

                feature = encoder_features[i]
                x = torch.cat([x, feature], dim=1)

            x = self.decode_blocks[i](x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        encoder_channels=[
            1,
            32,
            64,
            128,
            256,
            512,
        ],  # 1,32,64,128,256,512 # 1,64,128,256
        decoder_channels=[512, 256, 128, 64, 32],  # 256,128,64
        output_channel=2,
        out_activation_function=None,  # "tanh"
        skip=[],
    ):
        super().__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels, skip=skip)
        self.head = nn.Conv2d(decoder_channels[-1], output_channel, 1)
        if out_activation_function == "sigmoid":
            self.head = nn.Sequential(self.head, nn.Sigmoid())
        elif out_activation_function == "tanh":
            self.head = nn.Sequential(self.head, nn.Tanh())

    def identity(self, x):
        return x

    def forward(self, x):
        features = self.encoder(x)
        features.reverse()
        x = self.decoder(features[0], features[1:])
        x = self.head(x)
        return x


class skip_Decoder(nn.Module):
    def __init__(self, channels=[1024, 512, 256, 128, 64], skip=[], kernel_size=3):
        super().__init__()
        self.channels = channels
        self.skip = skip

        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                for i in range(len(channels) - 1)
            ]  # kernel_size used to be 4
        )
        self.decode_blocks = []
        self.skip_blocks = []
        for i in range(len(channels) - 1):
            if i in self.skip:
                self.decode_blocks.append(Block(2 * channels[i + 1], channels[i + 1]))
                self.skip_blocks.append(Block(channels[i + 1], channels[i + 1]))
            else:
                self.decode_blocks.append(Block(channels[i + 1], channels[i + 1]))
        self.decode_blocks = nn.ModuleList(self.decode_blocks)
        self.skip_blocks = nn.ModuleList(self.skip_blocks)

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            if i in self.skip:
                feature = encoder_features[i]
                feature = self.skip_blocks[i](feature)
                x = torch.cat([x, feature], dim=1)

            x = self.decode_blocks[i](x)
        return x


class skip_UNet(nn.Module):
    def __init__(
        self,
        encoder_channels=[
            1,
            32,
            64,
            128,
            256,
            512,
        ],  # 1,32,64,128,256,512 # 1,64,128,256
        decoder_channels=[512, 256, 128, 64, 32],  # 256,128,64
        output_channel=2,
        out_activation_function=None,  # "tanh"
        skip=[],
        kernel_size=3,
    ):
        super().__init__()
        self.encoder = Encoder(encoder_channels, kernel_size)
        self.decoder = skip_Decoder(decoder_channels, skip=skip)
        self.head = nn.Conv2d(decoder_channels[-1], output_channel, 1)
        if out_activation_function == "sigmoid":
            self.head = nn.Sequential(self.head, nn.Sigmoid())
        elif out_activation_function == "tanh":
            self.head = nn.Sequential(self.head, nn.Tanh())
        elif out_activation_function == "relu":
            self.head = nn.Sequential(self.head, nn.ReLU())
        elif out_activation_function == "leakyrelu":
            self.head = nn.Sequential(self.head, nn.LeakyReLU())

    def identity(self, x):
        return x

    def forward(self, x):
        features = self.encoder(x)
        features.reverse()
        x = self.decoder(features[0], features[1:])
        x = self.head(x)
        return x
