import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """
    Convolution Layer followed by optional Instance/Batch Normalization.
    Maintains exact attribute names for compatibility with existing checkpoints.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_type='instance', bias=True, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=kernel_size // 2, padding_mode='reflect', bias=bias, groups=groups
        )
        self.norm_type = norm_type
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        y = self.conv(x)
        if self.norm_type != 'None' and hasattr(self, 'norm'):
            y = self.norm(y)
        return y


class ResidualBlock(nn.Module):
    """
    Residual Block with two ConvLayers and a skip connection.
    Reference: https://arxiv.org/abs/1512.03385
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)

    def forward(self, x):
        residual = x
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        y = y + residual
        y = F.relu(y, inplace=True)
        return y


class UpsampleConvLayer(nn.Module):
    """
    Upsampling followed by convolution, avoiding checkerboard artifacts
    compared to transposed convolutions.
    Reference: http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, norm_type='instance'):
        super().__init__()
        self.upsample = upsample
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1, padding_mode='reflect')
        self.norm_type = norm_type
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=self.upsample)
        out = self.conv2d(x)
        if self.norm_type != 'None' and hasattr(self, 'norm'):
            out = self.norm(out)
        return out


class DeconvLayer(nn.Module):
    """
    Transposed convolution layer used in legacy Autoencoder architectures.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super().__init__()
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding_size, output_padding
        )
        self.norm_type = norm
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.norm_type != "None" and hasattr(self, 'norm_layer'):
            x = self.norm_layer(x)
        return x


class NormReluConv(nn.Module):
    """
    Normalization -> ReLU -> Conv, primarily used for DenseNet blocks.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_type="instance", bias=True):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "instance":
            self.norm_layer = nn.InstanceNorm2d(in_channels, affine=True)
        elif norm_type == "batch":
            self.norm_layer = nn.BatchNorm2d(in_channels, affine=True)

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=kernel_size // 2, padding_mode='reflect', bias=bias
        )

    def forward(self, x):
        if self.norm_type != "None" and hasattr(self, 'norm_layer'):
            x = self.norm_layer(x)
        x = F.relu(x, inplace=True)
        x = self.conv_layer(x)
        return x
