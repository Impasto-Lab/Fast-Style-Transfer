import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.common import ConvLayer, ResidualBlock, UpsampleConvLayer


class SelfAttention(nn.Module):
    """
    Self-Attention module for convolutional feature maps (SAGAN-style).
    Reference: 'Self-Attention Generative Adversarial Networks' (Zhang et al., 2019)
    """
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, max(1, channels // 8), 1, bias=False)
        self.key = nn.Conv1d(channels, max(1, channels // 8), 1, bias=False)
        self.value = nn.Conv1d(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        n_pixels = height * width
        x_flat = x.view(batch_size, channels, n_pixels)

        # Projections
        f = self.query(x_flat)              # (B, C//8, N)
        g = self.key(x_flat)                # (B, C//8, N)
        h = self.value(x_flat)              # (B, C, N)

        # Attention map: (B, N, N) where (i, j) is correlation of query i with key j
        energy = torch.bmm(f.transpose(1, 2), g)  # (B, N, N)
        attention = self.softmax(energy)          # Softmax across key dimension

        # Aggregate values weighted by attention
        out = self.gamma * torch.bmm(h, attention.transpose(1, 2)) + x_flat
        return out.view(batch_size, channels, height, width).contiguous()


class AutoencoderAttention(nn.Module):
    """
    Autoencoder equipped with a Self-Attention layer at the bottleneck resolution.
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

        self.sa = SelfAttention(128)

        self.ResidualBlock = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
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
        y = self.sa(y)
        y = self.ResidualBlock(y)
        y = self.UpSampleBlock(y)
        return y
