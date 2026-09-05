from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models

# Statically define VggOutputs namedtuple to prevent dynamic class creation on every forward pass
VggOutputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])


class Vgg16(nn.Module):
    """
    Pretrained VGG-16 network used for perceptual loss calculation (Johnson et al., 2016).
    Slices the network at:
    - slice1: up to relu1_2 (layer 3)
    - slice2: up to relu2_2 (layer 8)
    - slice3: up to relu3_3 (layer 15)
    - slice4: up to relu4_3 (layer 22)
    """
    def __init__(self, requires_grad=False):
        super().__init__()
        # Load weights using modern torchvision enum or legacy fallback
        try:
            weights = models.VGG16_Weights.DEFAULT
            vgg_pretrained_features = models.vgg16(weights=weights).features
        except (AttributeError, TypeError):
            vgg_pretrained_features = models.vgg16(pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return VggOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
