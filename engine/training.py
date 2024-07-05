#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
from tqdm import tqdm
from engine.utils import set_seed
from torch.utils.data import DataLoader
from engine.utils import elastic_saltpepper
set_seed(10)

def save_checkpoint(
        state: dict,
        new_path: str
        ) -> None:
    """
    Saves the current state of the model, optimizer, scheduler, and loss lists to a checkpoint file.

    Args:
        state (dict): A dictionary containing the current state of the model, optimizer, scheduler, and loss lists.
        new_path (str): The directory path where the checkpoint file will be saved.

    Returns:
        None

    Description:
        This function saves a checkpoint of the current state during model training. The checkpoint includes the current epoch, model state_dict, optimizer state_dict, scheduler state_dict, and lists of training and validation losses. It saves the checkpoint as 'checkpoint.pth.tar' in the specified directory path.
    """
    torch.save(state, f"{new_path}/checkpoint.pth.tar")

def load_checkpoint(
        new_path: str
        ) -> dict:
    """
    Loads a previously saved checkpoint.

    Args:
        new_path (str): The directory path where the checkpoint file is saved.

    Returns:
        dict: A dictionary containing the loaded checkpoint information.

    Description:
        This function loads a previously saved checkpoint file ('checkpoint.pth.tar') from the specified directory path. It returns a dictionary containing the epoch, model state_dict, optimizer state_dict, scheduler state_dict, and lists of training and validation losses stored in the checkpoint.
    """
    return torch.load(f"{new_path}/checkpoint.pth.tar")

def network_training(
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module, 
        scheduler: torch.optim.lr_scheduler,
        start_epoch: int, 
        num_epochs: int, 
        trainloader: DataLoader, 
        validationloader: DataLoader, 
        accumulation_steps: int, 
        device: torch.device, 
        new_path: str, 
        loss_list: list, 
        val_loss_list: list
        ) -> tuple:
    """
    Executes the training loop for a neural network model.

    Args:
        net (torch.nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function criterion.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        start_epoch (int): The starting epoch number.
        num_epochs (int): The total number of epochs for training.
        trainloader (DataLoader): DataLoader for the training dataset.
        validationloader (DataLoader): DataLoader for the validation dataset.
        accumulation_steps (int): Number of steps to accumulate gradients before optimization.
        device (torch.device): The device (GPU or CPU) to run the training on.
        new_path (str): The directory path where training outputs are saved.
        loss_list (list): List to store training losses.
        val_loss_list (list): List to store validation losses.

    Returns:
        tuple: A tuple containing updated loss lists and the trained neural network model.

    Description:
        This function performs the training loop for a neural network model. It iterates through each epoch, performing forward and backward passes on the training data, optimizing the model parameters, and evaluating the model on the validation set. It updates the learning rate based on validation loss using the scheduler and saves a checkpoint after each epoch. It returns the updated loss lists and the trained model.
    """

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