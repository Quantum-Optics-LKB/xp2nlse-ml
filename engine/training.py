#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
from tqdm import tqdm
from engine.utils import set_seed
from torch.utils.data import DataLoader
from engine.utils import elastic_saltpepper
from engine.network_dataset import NetworkDataset
from engine.engine_dataset import EngineDataset
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
        model_settings: tuple,
        dataset: EngineDataset,
        training_set: NetworkDataset,
        validation_set: NetworkDataset,
        test_set: NetworkDataset,
        loss_list: list, 
        val_loss_list: list
        ) -> tuple:
    
    model, optimizer, criterion, scheduler, device, new_path, start_epoch = model_settings
    
    training_loader = DataLoader(training_set, batch_size=dataset.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=dataset.batch_size, shuffle=True)

    # augment = elastic_saltpepper()
    for epoch in tqdm(range(start_epoch, dataset.num_epochs),desc=f"Training", 
                                total=dataset.num_epochs - start_epoch, unit="Epoch"):  
        running_loss = 0.0
        model.train()
        
        for i, (images, n2_labels, isat_labels, alpha_labels) in enumerate(training_loader, 0):            
            images = images.to(device = device, dtype=torch.float64)
            # dummy_channel = torch.ones(images.shape[0], 1, images.shape[2], images.shape[3],dtype=torch.float64)
            # images = augment(torch.cat((images, dummy_channel), dim=1))[:,0:2,:,:]

            n2_labels = n2_labels.to(device = device, dtype=torch.float64)
            isat_labels = isat_labels.to(device = device, dtype=torch.float64)
            alpha_labels = alpha_labels.to(device = device, dtype=torch.float64)
            
            labels = torch.cat((n2_labels,isat_labels, alpha_labels), dim=1)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()

            if (i + 1) % dataset.accumulator == 0 or dataset.accumulator == 1:
                optimizer.step()
                optimizer.zero_grad()  # Clear gradients after updating weights

            running_loss += loss.item()
        
        # Validation loop
        val_running_loss = 0.0
        model.eval()
        
        with torch.no_grad(): 
            for images, n2_labels, isat_labels, alpha_labels in validation_loader:
                images = images.to(device = device, dtype=torch.float64)
                n2_labels = n2_labels.to(device = device, dtype=torch.float64)
                isat_labels = isat_labels.to(device = device, dtype=torch.float64)
                alpha_labels = alpha_labels.to(device = device, dtype=torch.float64)

                labels = torch.cat((n2_labels,isat_labels, alpha_labels), dim=1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(validation_loader)
        scheduler.step(avg_val_loss)

        current_lr = scheduler.get_last_lr()
        print(f'Epoch {epoch+1}, Train Loss: {running_loss / len(training_loader)}, Validation Loss: {avg_val_loss}, Current LR: {current_lr[0]}')

        loss_list.append(running_loss / len(training_loader))
        val_loss_list.append(avg_val_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss_list': loss_list,
            'val_loss_list': val_loss_list
        },new_path)
    
    return loss_list, val_loss_list, model