#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
import torch.optim 

def network_training(net, optimizer, criterion, scheduler, num_epochs, trainloader, validationloader, device, accumulation_steps, backend):
    """
    Trains a neural network using given data loaders, optimizer, loss functions, and a learning rate scheduler.
    Supports gradient accumulation and device-specific data handling for efficient training.

    Parameters:
    - net (torch.nn.Module): The neural network model to be trained.
    - optimizer (torch.optim.Optimizer): The optimizer to use for adjusting model parameters.
    - criterion (list[torch.nn.Module]): A list of loss functions. The first is used for n2 label predictions,
      and the second for power label predictions.
    - scheduler (torch.optim.lr_scheduler): A learning rate scheduler to adjust the learning rate based on
      validation loss.
    - num_epochs (int): The number of epochs to train the model.
    - trainloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - validationloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - device (torch.device): The device (CPU/GPU) on which the training is executed.
    - accumulation_steps (int): The number of steps over which gradients are accumulated. Zero means that
      accumulation is not used.
    - backend (str): Indicates whether training is performed on CPU or GPU. Expected values are "CPU" or "GPU".

    Returns:
    tuple: Contains three elements:
        - loss_list (np.ndarray): Array of training loss values for each epoch.
        - val_loss_list (np.ndarray): Array of validation loss values for each epoch.
        - net (torch.nn.Module): The trained neural network model.

    This function orchestrates the training and validation process for a neural network model, implementing
    gradient accumulation if required, and performing device-specific data handling for optimized training.
    It tracks and prints the loss per epoch for both training and validation phases and returns the trained model
    along with the loss metrics.

    Example Usage:
        trained_model, training_losses, validation_losses = network_training(
            net=my_model, optimizer=my_optimizer, criterion=[loss_n2, loss_powers],
            scheduler=my_scheduler, num_epochs=20, trainloader=train_loader,
            validationloader=val_loader, device=torch.device('cuda'), accumulation_steps=4, backend="GPU"
        )
    """
    loss_list = np.zeros(num_epochs)
    val_loss_list = np.zeros(num_epochs)

    for epoch in range(num_epochs):  
        running_loss = 0.0
        net.train()

        for i, (images, power_values, power_labels, n2_labels, isat_labels) in enumerate(trainloader, 0):
            # Process original images
            if backend == "GPU":
                images = images
                power_labels = power_labels
                n2_labels = n2_labels
                isat_labels = isat_labels
                power_values = torch.from_numpy(power_values.cpu().numpy()[:,np.newaxis]).float().to(device)
            else:
                images = images.to(device) 
                power_labels = power_labels.to(device) 
                n2_labels = n2_labels.to(device)
                isat_labels = isat_labels.to(device)
                power_values = torch.from_numpy(power_values.numpy()[:,np.newaxis]).float().to(device)
                

            # # Forward pass with original images
            outputs_n2, outputs_power, outputs_isat = net(images, power_values)

            loss_n2 = criterion(outputs_n2, n2_labels)
            loss_powers = criterion(outputs_power, power_labels)
            loss_isat = criterion(outputs_isat, isat_labels)

            loss_n2.backward(retain_graph = True)
            loss_isat.backward(retain_graph = True)
            loss_powers.backward(retain_graph = True)

            if accumulation_steps == 0:
                optimizer.step()
            else:
                if (i + 1) % accumulation_steps == 0:
                    for param in net.parameters():
                        param.grad /= accumulation_steps

                    optimizer.step()

                    for param in net.parameters():
                        param.grad.zero_()
            
            running_loss += loss_n2.item() + loss_powers.item() + loss_isat.item()
        
        # Validation loop
        val_running_loss = 0.0
        net.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No gradients tracked
            for i, (images, power_values, power_labels, n2_labels, isat_labels) in enumerate(validationloader, 0):
                if backend == "GPU":
                    images = images
                    power_labels = power_labels
                    n2_labels = n2_labels
                    isat_labels = isat_labels
                    power_values = torch.from_numpy(power_values.cpu().numpy()[:,np.newaxis]).float().to(device)
                    
                else:
                    images = images.to(device) 
                    power_labels = power_labels.to(device) 
                    n2_labels = n2_labels.to(device)
                    isat_labels = isat_labels.to(device)
                    power_values = torch.from_numpy(power_values.numpy()[:,np.newaxis]).float().to(device)

                # Forward pass with original images
                outputs_n2, outputs_isat = net(images, power_values)

                loss_n2 = criterion(outputs_n2, n2_labels)
                loss_powers = criterion(outputs_power, power_labels)
                loss_isat = criterion(outputs_isat, isat_labels)
                
                
                val_running_loss += loss_n2.item() + loss_powers.item() + loss_isat.item()

        # Step the scheduler with the validation loss
        scheduler.step(val_running_loss / len(validationloader))

        # Print the epoch's result
        print(f'Epoch {epoch+1}, Train Loss: {running_loss / len(trainloader)}, Validation Loss: {val_running_loss / len(validationloader)}')
        # print(f'Epoch {epoch+1}, Train accuracy: {np.exp(-running_loss / len(trainloader)) *100} %, Validation accuracy: {np.exp(-val_running_loss / len(validationloader))*100} %')

        loss_list[epoch] = running_loss / len(trainloader)
        val_loss_list[epoch] = val_running_loss / len(validationloader)
    
    return loss_list, val_loss_list, net