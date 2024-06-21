#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import random
import torch
import numpy as np
import torch.optim
from tqdm import tqdm
import kornia.augmentation as K

def augmentation(
        original_height: int, 
        original_width: int
        ) -> torch.nn.Sequential:
    """
    Constructs a data augmentation pipeline with a specific set of transformations tailored for 
    optical field data or similar types of images. This pipeline includes blurring, cropping, and 
    resizing operations to simulate various realistic alterations that data might undergo.

    Parameters:
    - original_height (int): The original height of the images before augmentation.
    - original_width (int): The original width of the images before augmentation.

    Returns:
    torch.nn.Sequential: A Kornia Sequential object that contains a 
    sequence of augmentation transformations to be applied to the images. These transformations 
    include Gaussian blur, motion blur, a slight shift without scaling or rotation, 
    and resizing back to the original dimensions.

    The pipeline is set up to apply these transformations with certain probabilities, allowing for a 
    diversified dataset without excessively distorting the underlying data characteristics. This 
    augmentation strategy is particularly useful for training machine learning models on image data, 
    as it helps to improve model robustness by exposing it to a variety of visual perturbations.

    Example Usage:
        augmentation_pipeline = get_augmentation(256, 256)
        augmented_image = augmentation_pipeline(image=image)
    """
    shift = random.uniform(0.1,0.25)
    shear = random.uniform(20,50)
    direction = random.uniform(-1, 1)
    return torch.nn.Sequential(
        K.RandomMotionBlur(kernel_size=51, angle=random.uniform(0, 360), direction=(direction, direction), border_type='replicate', p=0.2),
        K.RandomGaussianBlur(kernel_size=(51, 51), sigma=(100.0, 100.0), p=0.5),
        K.RandomAffine(degrees=0, translate=(shift, shift), scale=(1.0, 1.0), shear=shear, p=0.2),
        K.Resize((original_height, original_width))
    )

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