import os
import time

from typing import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from models import *


# Replacing MobileNetV2's relu6 with relu to allow
# usage of CoreML on iOS devices.
def replace_relu6_with_relu(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_relu6_with_relu(model=module)
        if isinstance(module, nn.ReLU6):
            model._modules[name] = nn.ReLU()
    return model

dwsepconv = lambda inchannels, outchannels, kernels: nn.Sequential(
    depthwise(inchannels, kernels),
    pointwise(inchannels, outchannels),
)


# We also need to replace Mobilenet's ReLU6 activations with ReLU. 
# There is no noticeable difference in quality, but this will
# allow us to use CoreML for mobile inference on iOS devices.
def replace_relu6_with_relu(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_relu6_with_relu(model=module)
        if isinstance(module, nn.ReLU6):
            model._modules[name] = nn.ReLU()
    return model

# Model architecture adapted from: https://github.com/Snapchat/snapml-templates
class AnimeNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2(width_mult=0.5)

        # We reuse state dict from mobilenet v2 width width_mult == 1.0.
        # This is not the optimal way to use pretrained models, but in this case
        # it gives us good initialization for faster convergence.
        state_dict = torchvision.models.mobilenet_v2(pretrained=True).state_dict()
        target_dict = mobilenet.state_dict()
        for k in target_dict.keys():
            if len(target_dict[k].size()) == 0:
                continue
            state_dict[k] = state_dict[k][:target_dict[k].size(0)]
            if len(state_dict[k].size()) > 1:
                state_dict[k] = state_dict[k][:, :target_dict[k].size(1)]

        mobilenet.load_state_dict(state_dict)

        weight = mobilenet.features[0][0].weight.detach()
        # mobilenet.features[0][0].weight = nn.Parameter(data=weight / 255.)

        mobilenet = replace_relu6_with_relu(mobilenet)

        self.features = mobilenet.features[:-2]
        self.upscale0 = nn.Sequential(
            nn.Conv2d(80, 48, 1, 1, 0, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.upscale1 = nn.Sequential(
            nn.Conv2d(48, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.upscale2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.upscale3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.upscale4 = nn.Sequential(
            nn.Conv2d(8, 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.upscale5 = nn.Conv2d(4, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        out = x
        skip_outs = []
        for i in range(len(self.features)):
            out = self.features[i](out)
            if i in {1, 3, 6, 13}:
                skip_outs.append(out)
        out = self.upscale0(out)
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale1(out + skip_outs[3])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale2(out + skip_outs[2])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale3(out + skip_outs[1])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale4(out + skip_outs[0])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale5(out)
        return torch.sigmoid(out)


# This model is a combination of segmentation model from: https://github.com/Snapchat/snapml-templates
# and the fast-depth model from: https://github.com/dwofk/fast-depth
# This model is NOT used in the project. It was ruled out early because it
# offered no significant learning or size advantage over the above model. It
# is included for completeness.
class AnimeNNConv5Net(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v2(width_mult=0.5)

        # We reuse state dict from mobilenet v2 width width_mult == 1.0.
        # This is not the optimal way to use pretrained models, but in this case
        # it gives us good initialization for faster convergence.
        state_dict = torchvision.models.mobilenet_v2(pretrained=True).state_dict()
        target_dict = mobilenet.state_dict()
        for k in target_dict.keys():
            if len(target_dict[k].size()) == 0:
                continue
            state_dict[k] = state_dict[k][:target_dict[k].size(0)]
            if len(state_dict[k].size()) > 1:
                state_dict[k] = state_dict[k][:, :target_dict[k].size(1)]

        mobilenet.load_state_dict(state_dict)

        weight = mobilenet.features[0][0].weight.detach()
        # mobilenet.features[0][0].weight = nn.Parameter(data=weight / 255.)

        mobilenet = replace_relu6_with_relu(mobilenet)

        self.features = mobilenet.features[:-2]
        self.upscale0 = dwsepconv(80, 48, 5)
        self.upscale1 = dwsepconv(48, 16, 5)
        self.upscale2 = dwsepconv(16, 16, 3)
        self.upscale3 = dwsepconv(16, 8, 5)
        self.upscale4 = dwsepconv(8, 4, 5)
        self.upscale5 = dwsepconv(4, 1, 5)

    def forward(self, x):
        out = x
        skip_outs = []
        for i in range(len(self.features)):
            out = self.features[i](out)
            if i == 1 or i == 3 or i == 6 or i == 13:
                skip_outs.append(out)

        out = self.upscale0(out)
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale1(out + skip_outs[3])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale2(out + skip_outs[2])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale3(out + skip_outs[1])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale4(out + skip_outs[0])
        out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.upscale5(out)
        return out
