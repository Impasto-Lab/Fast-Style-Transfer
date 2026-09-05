import torch
import torch.nn as nn
from src.models.common import ConvLayer, ResidualBlock, DeconvLayer


class AutoencoderOld(nn.Module):
    """
    Legacy Autoencoder architecture using transposed convolutions instead of upsampling convolutions.
    """
    def __init__(self):
        super().__init__()
        # Initial downsampling convolution block
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(inplace=True),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(inplace=True),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU(inplace=True)
        )

        # Residual bottleneck blocks
        self.ResidualBlock = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )

        # Transposed deconvolution block
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            ConvLayer(32, 3, 9, 1, norm_type='None')
        )

    def forward(self, x):
        y = self.ConvBlock(x)
        y = self.ResidualBlock(y)
        y = self.DeconvBlock(y)
        return y
