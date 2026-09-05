import torch
import torch.nn as nn
from src.models.common import ConvLayer, ResidualBlock, UpsampleConvLayer


class Autoencoder(nn.Module):
    """
    Standard Feedforward Image Transformation Network from:
    'Perceptual Losses for Real-Time Style Transfer and Super-Resolution' (Johnson et al., 2016)
    Uses reflection padding and nearest-neighbor upsampling followed by convolutions.
    """
    def __init__(self):
        super().__init__()
        # Initial downsampling convolution block
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),   # (b, 32, h, w)
            nn.ReLU(inplace=True),
            ConvLayer(32, 64, 3, 2),   # (b, 64, h//2, w//2)
            nn.ReLU(inplace=True),
            ConvLayer(64, 128, 3, 2),  # (b, 128, h//4, w//4)
            nn.ReLU(inplace=True)
        )

        # Residual bottleneck blocks
        self.ResidualBlock = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )

        # Upsampling convolution block
        self.UpSampleBlock = nn.Sequential(
            UpsampleConvLayer(128, 64, 3, 1, 2),  # (b, 64, h//4, w//4)
            nn.ReLU(inplace=True),
            UpsampleConvLayer(64, 32, 3, 1, 2),   # (b, 32, h//2, w//2)
            nn.ReLU(inplace=True),
            ConvLayer(32, 3, 9, 1, norm_type='None')  # (b, 3, h, w)
        )

    def forward(self, x):
        y = self.ConvBlock(x)
        y = self.ResidualBlock(y)
        y = self.UpSampleBlock(y)
        return y
