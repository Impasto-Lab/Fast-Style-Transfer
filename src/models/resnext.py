import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.common import ConvLayer, UpsampleConvLayer


class ResNextBottleneck(nn.Module):
    """
    ResNeXt Bottleneck Block with grouped convolutions.
    Reference: 'Aggregated Residual Transformations for Deep Neural Networks' (Xie et al., 2017)
    https://arxiv.org/abs/1611.05431
    """
    def __init__(self, channels, cardinality=32):
        super().__init__()
        mid_channels = channels // 2
        self.conv1 = ConvLayer(channels, mid_channels, 1, 1, norm_type='batch', bias=False)
        self.conv2 = ConvLayer(mid_channels, mid_channels, 3, 1, groups=cardinality, norm_type='batch', bias=False)
        self.conv3 = ConvLayer(mid_channels, channels, 1, 1, norm_type='batch', bias=False)

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y, inplace=True)
        y = self.conv2(y)
        y = F.relu(y, inplace=True)
        y = self.conv3(y)
        return F.relu(x + y, inplace=True)


class ResNext(nn.Module):
    """
    Style transfer generator using ResNeXt grouped bottleneck blocks.
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

        self.ResNextBottleneck = nn.Sequential(
            *[ResNextBottleneck(128, 32) for _ in range(5)]
        )

        self.UpSampleBlock = nn.Sequential(
            UpsampleConvLayer(128, 64, 3, 1, 2),
            nn.ReLU(inplace=True),
            UpsampleConvLayer(64, 32, 3, 1, 2),
            nn.ReLU(inplace=True),
            ConvLayer(32, 3, 9, 1, norm_type='None')
        )

    def forward(self, x):
        y = self.ConvBlock(x)
        y = self.ResNextBottleneck(y)
        y = self.UpSampleBlock(y)
        return y
