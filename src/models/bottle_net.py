import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.common import ConvLayer, UpsampleConvLayer


class BottleNetLayer(nn.Module):
    """
    Bottleneck residual layer (1x1 -> 3x3 -> 1x1 convs)
    Reference: https://arxiv.org/abs/1611.05431
    """
    def __init__(self, in_ch=128, channels=[64, 64, 128], kernel_size=3):
        super().__init__()
        ch1, ch2, ch3 = channels
        self.conv1 = ConvLayer(in_ch, ch1, kernel_size=1, stride=1)
        self.conv2 = ConvLayer(ch1, ch2, kernel_size=kernel_size, stride=1)
        self.conv3 = ConvLayer(ch2, ch3, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        out = self.conv3(out)
        out = out + identity
        return out


class BottleNetwork(nn.Module):
    """
    Style transfer generator using Bottleneck residual blocks.
    """
    def __init__(self):
        super().__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(inplace=True),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(inplace=True),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU(inplace=True)
        )

        self.ResidualBlock = nn.Sequential(
            *[BottleNetLayer(128, [64, 64, 128], kernel_size=3) for _ in range(5)]
        )

        self.UpSampleBlock = nn.Sequential(
            UpsampleConvLayer(128, 64, 3, 1, 2),
            nn.ReLU(inplace=True),
            UpsampleConvLayer(64, 32, 3, 1, 2),
            nn.ReLU(inplace=True),
            ConvLayer(32, 3, 9, 1, norm_type='None')
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.UpSampleBlock(x)
        return out
