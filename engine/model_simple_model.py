#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
# inspired from https://github.com/zhulf0804/Inceptionv4_and_Inception-ResNetv2.PyTorch/tree/master

import torch
import torch.nn as nn
from engine.utils import set_seed
set_seed(10)

class Inception_ResNetv2(nn.Module):
    def __init__(self, in_channels=2):
        super(Inception_ResNetv2, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Adjust the input size here based on pooling
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)  # Output a triplet of parameters
        
        # Activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Apply conv layers with ReLU and max pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        
        # Flatten the output from conv layers
        x = x.view(-1, 256 * 14 * 14)
        
        # Fully connected layers with ReLU and dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (no activation for regression)
        
        return x