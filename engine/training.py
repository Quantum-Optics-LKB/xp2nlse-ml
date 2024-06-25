#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import random
import torch
import numpy as np
import torch.optim
from tqdm import tqdm
from engine.noise_generator import augmentation
from engine.seed_settings import set_seed
set_seed(10)

def network_training(net, optimizer, criterion, scheduler, num_epochs, trainloader, validationloader, accumulation_steps, device):

    loss_list = np.zeros(num_epochs)
    val_loss_list = np.zeros(num_epochs)

    for epoch in tqdm(range(num_epochs),desc=f"Training", 
                                total=num_epochs, unit="Epoch"):  
        running_loss = 0.0
        net.train()
        
        for i, (images, n2_values, isat_values) in enumerate(trainloader, 0):
            augment = augmentation(images.shape[-2],images.shape[-1])
            
            images = augment(images.to(device = device, dtype=torch.float32))
            n2_values = n2_values.to(device = device, dtype=torch.float32)
            isat_values = isat_values.to(device = device, dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs_n2, outputs_isat = net(images)

            loss_n2 = criterion(outputs_n2, n2_values)
            loss_isat = criterion(outputs_isat, isat_values)
            loss_n2.backward(retain_graph=True)
            loss_isat.backward()

            if (i + 1) % accumulation_steps == 0 or accumulation_steps == 1:
                optimizer.step()
                optimizer.zero_grad()  # Clear gradients after updating weights

            running_loss += loss_n2.item() + loss_isat.item()
        
        # Validation loop
        val_running_loss = 0.0
        net.eval()
        with torch.no_grad(): 
            for images, n2_values, isat_values in validationloader:
                images = images.to(device = device, dtype=torch.float32)
                n2_values = n2_values.to(device = device, dtype=torch.float32)
                isat_values = isat_values.to(device = device, dtype=torch.float32)

                outputs_n2, outputs_isat = net(images)
                loss_n2 = criterion(outputs_n2, n2_values)
                loss_isat = criterion(outputs_isat, isat_values)
                
                val_running_loss += loss_n2.item() + loss_isat.item()

        avg_val_loss = val_running_loss / len(validationloader)
        scheduler.step(avg_val_loss)

        current_lr = scheduler.get_last_lr()  # Get current learning rate after update
        print(f'Epoch {epoch+1}, Train Loss: {running_loss / len(trainloader):.4f}, Validation Loss: {avg_val_loss:.4f}, Current LR: {current_lr[0]}')

        loss_list[epoch] = running_loss / len(trainloader)
        val_loss_list[epoch] = avg_val_loss
    
    return loss_list, val_loss_list, net