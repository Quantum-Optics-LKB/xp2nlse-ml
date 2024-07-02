#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from tqdm import tqdm
from engine.seed_settings import set_seed
from engine.treament import elastic_saltpepper
set_seed(10)

def save_checkpoint(state,new_path):
    torch.save(state, f"{new_path}/checkpoint.pth.tar")

def load_checkpoint(new_path):
    return torch.load(f"{new_path}/checkpoint.pth.tar")

def network_training(net,
                    optimizer, 
                    criterion, 
                    scheduler,
                    start_epoch, 
                    num_epochs, 
                    trainloader, 
                    validationloader, 
                    accumulation_steps, 
                    device, 
                    new_path, 
                    loss_list, 
                    val_loss_list):

    augment = elastic_saltpepper()
    for epoch in tqdm(range(start_epoch, num_epochs),desc=f"Training", 
                                total=num_epochs - start_epoch, unit="Epoch"):  
        running_loss = 0.0
        net.train()
        
        for i, (images, n2_values, isat_values, alpha_values) in enumerate(trainloader, 0):            
            images = augment(images.to(device = device, dtype=torch.float32))
            n2_values = n2_values.to(device = device, dtype=torch.float32)
            isat_values = isat_values.to(device = device, dtype=torch.float32)
            alpha_values = alpha_values.to(device = device, dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs_n2, outputs_isat, outputs_alpha = net(images)

            loss_n2 = criterion(outputs_n2, n2_values)
            loss_isat = criterion(outputs_isat, isat_values)
            loss_alpha = criterion(outputs_alpha, alpha_values)
            loss_n2.backward(retain_graph=True)
            loss_isat.backward(retain_graph=True)
            loss_alpha.backward()

            if (i + 1) % accumulation_steps == 0 or accumulation_steps == 1:
                optimizer.step()
                optimizer.zero_grad()  # Clear gradients after updating weights

            running_loss += loss_n2.item() + loss_isat.item()
        
        # Validation loop
        val_running_loss = 0.0
        net.eval()
        with torch.no_grad(): 
            for images, n2_values, isat_values, alpha_values in validationloader:
                images = images.to(device = device, dtype=torch.float32)
                n2_values = n2_values.to(device = device, dtype=torch.float32)
                isat_values = isat_values.to(device = device, dtype=torch.float32)
                alpha_values = alpha_values.to(device = device, dtype=torch.float32)

                outputs_n2, outputs_isat, outputs_alpha = net(images)
                loss_n2 = criterion(outputs_n2, n2_values)
                loss_isat = criterion(outputs_isat, isat_values)
                loss_alpha = criterion(outputs_alpha, alpha_values)
                
                val_running_loss += loss_n2.item() + loss_isat.item() + loss_alpha.item()

        avg_val_loss = val_running_loss / len(validationloader)
        scheduler.step(avg_val_loss)

        current_lr = scheduler.get_last_lr()  # Get current learning rate after update
        print(f'Epoch {epoch+1}, Train Loss: {running_loss / len(trainloader):.4f}, Validation Loss: {avg_val_loss:.4f}, Current LR: {current_lr[0]}')

        loss_list.append(running_loss / len(trainloader))
        val_loss_list.append(avg_val_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss_list': loss_list,
            'val_loss_list': val_loss_list
        },new_path)
    
    return loss_list, val_loss_list, net