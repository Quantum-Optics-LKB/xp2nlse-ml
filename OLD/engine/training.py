#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
from tqdm import tqdm
from engine.utils import set_seed
from torch.utils.data import DataLoader
from engine.utils import elastic_saltpepper
from engine.field_dataset import FieldDataset
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
        fieldset: FieldDataset, 
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
    fieldset.flag = "training"
    training_loader = DataLoader(fieldset, batch_size=fieldset.batch_size, shuffle=True)
    fieldset.flag = "validation"
    validation_loader = DataLoader(fieldset, batch_size=fieldset.batch_size, shuffle=True)

    augment = elastic_saltpepper()
    for epoch in tqdm(range(start_epoch, num_epochs),desc=f"Training", 
                                total=num_epochs - start_epoch, unit="Epoch"):  
        running_loss = 0.0
        net.train()
    
        
        for i, (images, n2_values, isat_values, alpha_values) in enumerate(training_loader, 0):            
            images = images.to(device = device, dtype=torch.float64)
            dummy_channel = torch.ones(images.shape[0], 1, images.shape[2], images.shape[3],dtype=torch.float64)
            images = augment(torch.cat((images, dummy_channel), dim=1))[:,0:2,:,:]

            n2_values = n2_values.to(device = device, dtype=torch.float64)
            isat_values = isat_values.to(device = device, dtype=torch.float64)
            alpha_values = alpha_values.to(device = device, dtype=torch.float64)
            
            values = torch.cat((n2_values,isat_values, alpha_values), dim=1)

            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, values)
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or accumulation_steps == 1:
                optimizer.step()
                optimizer.zero_grad()  # Clear gradients after updating weights

            running_loss += loss.item()
        
        # Validation loop
        val_running_loss = 0.0
        net.eval()
        
        with torch.no_grad(): 
            for images, n2_values, isat_values, alpha_values in validation_loader:
                images = images.to(device = device, dtype=torch.float64)
                n2_values = n2_values.to(device = device, dtype=torch.float64)
                isat_values = isat_values.to(device = device, dtype=torch.float64)
                alpha_values = alpha_values.to(device = device, dtype=torch.float64)

                values = torch.cat((n2_values,isat_values, alpha_values), dim=1)

                outputs = net(images)
                loss = criterion(outputs, values)
                
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(validation_loader)
        scheduler.step(avg_val_loss)

        current_lr = scheduler.get_last_lr()
        print(f'Epoch {epoch+1}, Train Loss: {running_loss / len(training_loader)}, Validation Loss: {avg_val_loss}, Current LR: {current_lr[0]}')

        loss_list.append(running_loss / len(training_loader))
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