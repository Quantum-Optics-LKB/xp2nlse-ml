#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
# inspired from https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch/tree/master

import torch
import torch.nn as nn
import timm
from engine.utils import set_seed
set_seed(10)

class SingleChannelInceptionResNetV2(nn.Module):
    def __init__(self):
        super(SingleChannelInceptionResNetV2, self).__init__()
        # Load the Inception-ResNet v2 model
        self.model = timm.create_model('inception_resnet_v2', pretrained=False)
        
        # Modify the first convolutional layer to accept single-channel input
        self.model.conv2d_1a.conv = nn.Conv2d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=3, 
            stride=2, 
            bias=False
        )
        
        # Adjust the classifier to output features instead of classification
        self.model.classif = nn.Identity()
        
    def forward(self, x):
        x = self.model(x)
        return x

# Main model combining both branches
class Inception_ResNetv2(nn.Module):
    def __init__(self, in_channels=2):
        super(Inception_ResNetv2, self).__init__()
        # Separate models for density and phase images
        self.density_model = SingleChannelInceptionResNetV2()
        self.phase_model = SingleChannelInceptionResNetV2()
        
        # Fully connected layers after concatenation
        self.fc = nn.Sequential(
            nn.Linear(1536 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)  # Output layer for regression
        )
        
    def forward(self, input):
        density_input = input[:,[0],:,:]
        phase_input = input[:,[1],:,:]
        # Forward pass through density model
        density_features = self.density_model(density_input)
        # Forward pass through phase model
        phase_features = self.phase_model(phase_input)
        
        # Flatten features
        density_features = torch.flatten(density_features, 1)
        phase_features = torch.flatten(phase_features, 1)
        
        # Concatenate features
        combined_features = torch.cat((density_features, phase_features), dim=1)
        
        # Fully connected layers
        output = self.fc(combined_features)
        return output