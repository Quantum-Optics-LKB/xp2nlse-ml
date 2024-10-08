#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import torch.nn as nn
import torchvision.models as models
from engine.utils import set_seed
set_seed(10)

class SingleChannelResNet50(nn.Module):
    def __init__(self):
        super(SingleChannelResNet50, self).__init__()
        self.model = models.resnet18()
        
        self.model.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.model.fc = nn.Identity()
        
    def forward(self, x):
        x = self.model(x)
        return x

class Inception_ResNetv2(nn.Module):
    def __init__(self, in_channels=2):
        super(Inception_ResNetv2, self).__init__()

        self.resnet = SingleChannelResNet50()
        
        # nn.Linear(2048, 1024),
        # nn.ReLU(),
        # nn.Dropout(0.1),
        # nn.Linear(1024, 512),
        # nn.ReLU(),
        # nn.Dropout(0.1),
        self.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(64, 3),
        nn.Sigmoid()
    )
        
    def forward(self, input):

        features = self.resnet(input)
                
        output = self.fc(features)
        
        return output
