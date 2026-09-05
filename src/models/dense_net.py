import torch
import torch.nn as nn
from src.models.common import ConvLayer, NormReluConv, UpsampleConvLayer


class DenseLayerBottleNeck(nn.Module):
    """
    Dense layer with bottleneck: Norm -> ReLU -> Conv 1x1 -> Norm -> ReLU -> Conv 3x3,
    concatenating input along channel dimension.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = NormReluConv(in_channels, 4 * out_channels, 1, 1, bias=False)
        self.conv3 = NormReluConv(4 * out_channels, out_channels, 3, 1, bias=False)

    def forward(self, x):
        out = self.conv3(self.conv1(x))
        return torch.cat((x, out), dim=1)


class DenseNet(nn.Module):
    """
    Feedforward Transformer Network using DenseNet Blocks instead of Residual Blocks.
    """
    def __init__(self):
        super().__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1, bias=False),
            nn.ReLU(inplace=True),
            ConvLayer(32, 64, 3, 2, bias=False),
            nn.ReLU(inplace=True),
            ConvLayer(64, 128, 3, 2, bias=False),
            nn.ReLU(inplace=True)
        )

        self.DenseBlock = nn.Sequential(
            NormReluConv(128, 64, 1, 1),
            DenseLayerBottleNeck(64, 16),
            DenseLayerBottleNeck(80, 16),
            DenseLayerBottleNeck(96, 16),
            DenseLayerBottleNeck(112, 16)
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
        x = self.DenseBlock(x)
        out = self.UpSampleBlock(x)
        return out
