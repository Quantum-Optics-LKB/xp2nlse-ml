#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import torch.nn as nn
import torchvision.models as models
from engine.utils import set_seed
set_seed(10)
        
class Inception_ResNetv2(nn.Module):
    def __init__(self, in_channels=2):
        super(Inception_ResNetv2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 224 * 224, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

