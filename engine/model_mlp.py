#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import torch.nn as nn
from engine.utils import set_seed
set_seed(10)
        
class network(nn.Module):
    def __init__(self, 
                 input_channels=2, 
                 image_size=(224, 224), 
                 hidden_size=128, 
                 combined_hidden_size=256, 
                 output_size=3):
        super(network, self).__init__()
        
        self.input_channels = input_channels
        self.image_size = image_size
        self.output_size = output_size
        
        self.flatten_size = image_size[0] * image_size[1]
        
        self.channel_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.flatten_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ) for _ in range(input_channels)
        ])
        
        self.combined_layers = nn.Sequential(
            nn.Linear(hidden_size * input_channels, combined_hidden_size),
            nn.ReLU(),
            nn.Linear(combined_hidden_size, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        assert x.shape[1] == self.input_channels, f"Expected {self.input_channels} channels, got {x.shape[1]}"
        
        channel_outputs = []
        for i in range(self.input_channels):
            channel = x[:, i, :, :]  # (batch_size, H, W)
            channel = channel.view(x.size(0), -1)  # (batch_size, H*W)
            out = self.channel_nets[i](channel)  # (batch_size, hidden_size)
            channel_outputs.append(out)
        
        combined = torch.cat(channel_outputs, dim=1)  # (batch_size, hidden_size * 2)
        outputs = self.combined_layers(combined)      # (batch_size, 3)
        
        return outputs