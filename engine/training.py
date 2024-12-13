#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from tqdm import tqdm
from io import TextIOWrapper
import torch.nn.utils as nn_utils
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from engine.engine_dataset import EngineDataset
from engine.network_dataset import NetworkDataset
from engine.utils import augmentation_density, augmentation_phase, set_seed


set_seed(10)

def save_checkpoint(
        state: dict,
        new_path: str
        ) -> None:
    """
    Saves the training state to a checkpoint file.

    Parameters:
    -----------
    state : dict
        A dictionary containing the model state, optimizer state, scheduler state,
        training epoch, and loss lists.
    new_path : str
        Directory path to save the checkpoint file.

    Returns:
    --------
    None
        Saves the checkpoint as 'checkpoint.pth.tar' in the specified directory.
    """
    torch.save(state, f"{new_path}/checkpoint.pth.tar")

def load_checkpoint(
        new_path: str
        ) -> dict:
    """
    Loads a saved training state from a checkpoint file.

    Parameters:
    -----------
    new_path : str
        Directory path where the checkpoint file is stored.

    Returns:
    --------
    dict
        A dictionary containing the model state, optimizer state, scheduler state,
        training epoch, and loss lists from the checkpoint.
    """
    return torch.load(f"{new_path}/checkpoint.pth.tar", weights_only=True)


def network_training(
        model_settings: tuple,
        dataset: EngineDataset,
        training_set: NetworkDataset,
        validation_set: NetworkDataset,
        loss_list: list, 
        val_loss_list: list,
        file: TextIOWrapper,
        ) -> tuple:
        
    """
    Trains the neural network model with the specified settings and data.

    Parameters:
    -----------
    model_settings : tuple
        A tuple containing the model, optimizer, loss function, scheduler, device,
        save path, starting epoch, and loss threshold.
    dataset : EngineDataset
        The dataset containing simulation and training configurations.
    training_set : NetworkDataset
        Training dataset.
    validation_set : NetworkDataset
        Validation dataset.
    loss_list : list
        List to store training losses for each epoch.
    val_loss_list : list
        List to store validation losses for each epoch.
    file : TextIOWrapper
        A writable file object for logging results.

    Returns:
    --------
    tuple
        Updated `loss_list`, `val_loss_list`, and the trained model.
    """
    
    model, optimizer, criterion, scheduler, device, new_path, start_epoch, loss_threshold = model_settings
    
    # Initialize data loaders for training and validation datasets
    training_loader = DataLoader(training_set, batch_size=dataset.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=dataset.batch_size, shuffle=True)

    # Parameters for dynamic batch size reduction
    batch_reduction_factor = 4 
    best_val_loss = float('inf')
    patience = 20 # Early stopping patience
    trigger_times = 0

    # Randomize augmentation parameters
    base_shear = np.random.uniform(3, 7, 1)[0]
    shear = (base_shear, np.random.uniform(base_shear, 7, 1)[0])
    rotation_degrees = np.random.uniform(10, 25, 1)[0] 

    # Define augmentation pipelines
    augment_density = augmentation_density(rotation_degrees, shear)
    augment_phase = augmentation_phase(rotation_degrees, shear)

    progress_bar = tqdm(
        range(start_epoch, dataset.num_epochs),
        desc=f"Training", 
        total=dataset.num_epochs - start_epoch, 
        unit="Epoch"
        )
    
    for epoch in progress_bar:  
        running_loss = 0.0
        model.train()
        
        # Iterate through training batches
        for i, (images, n2_labels, isat_labels, alpha_labels) in enumerate(training_loader, 0):      
            
            images = images.to(device = device)
            images[:,0, :, :] = augment_density(images[:,0, :, :]).to(device = device)
            images[:,1, :, :] = augment_phase(images[:,1, :, :]).to(device = device)

            # Move labels to device
            n2_labels = n2_labels.to(device = device)
            isat_labels = isat_labels.to(device = device)
            alpha_labels = alpha_labels.to(device = device)   

            # Concatenate labels and compute loss
            labels = torch.cat((n2_labels, isat_labels, alpha_labels), dim=1)
            outputs, cov_outputs = model(images)
            loss = criterion(outputs, cov_outputs, labels)
            loss.backward()
            
            # Gradient accumulation
            if (i + 1) % dataset.accumulator == 0 or dataset.accumulator == 1:
                nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
        
        avg_run_loss = running_loss / len(training_loader)
        
        # Validation phase
        val_running_loss = 0.0
        mae_values = torch.zeros(3, dtype=torch.float32, device=device)
        all_preds = []
        all_labels = []

        model.eval()
        with torch.no_grad(): 
            for images, n2_labels, isat_labels, alpha_labels in validation_loader:
                images = images.to(device = device)
                n2_labels = n2_labels.to(device = device)
                isat_labels = isat_labels.to(device = device)
                alpha_labels = alpha_labels.to(device = device)
                labels = torch.cat((n2_labels, isat_labels, alpha_labels), dim=1)
            
                outputs, cov_outputs = model(images)
            
                loss = criterion(outputs, cov_outputs, labels)
                val_running_loss += loss.item()                

                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

                mae_values += torch.abs(outputs - labels).sum(axis=0)

        avg_val_loss = val_running_loss / len(validation_loader)
        mae_values /= len(validation_loader.dataset)

        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)

        # Compute R² score for each parameter
        r2_n2 = r2_score(all_labels_tensor[:,0].numpy(), all_preds_tensor[:,0].numpy())
        r2_isat = r2_score(all_labels_tensor[:,1].numpy(), all_preds_tensor[:,1].numpy())
        r2_alpha = r2_score(all_labels_tensor[:,2].numpy(), all_preds_tensor[:,2].numpy())
        average_r2 = (r2_n2 + r2_isat + r2_alpha) / 3

        scheduler.step(avg_val_loss)
        
        # Log performance metrics
        current_lr = scheduler.get_last_lr()
        record = f'Epoch {epoch}, Train Loss: {avg_run_loss},'
        record += f'Validation Loss: {avg_val_loss}, Current LR: {current_lr[0]}, R2_n2: {r2_n2},'
        record += f'R2_Isat: {r2_isat}, R2_alpha: {r2_alpha}, Average R2: {average_r2:.4f}, MAE: {mae_values.tolist()}'

        record_file = record + '\n'
        file.write(record_file)
        record_print = '\n' + record
        print(record_print, flush=True)

        if mae_values.tolist()[0] < loss_threshold:
            if dataset.batch_size == dataset.batch_size*dataset.accumulator:
                file.write(f"Batch size cannot be further reduced\n")
                print(f"Batch size cannot be further reduced", flush=True)

            else:
                dataset.accumulator = max(dataset.accumulator // batch_reduction_factor, 1)
                
                weight_decay =  1e-5
                optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=current_lr[0] / batch_reduction_factor, 
                    weight_decay=weight_decay
                    )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
                
                loss_threshold *= 0.9
                
                file.write(f"Batch size reduced to {int(dataset.batch_size*dataset.accumulator)}")
                print(f"Batch size reduced to {int(dataset.batch_size*dataset.accumulator)}", flush=True)
        
        loss_list.append(avg_run_loss)
        val_loss_list.append(avg_val_loss)
        
        # Save checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss_list': loss_list,
            'val_loss_list': val_loss_list,
            "loss_threshold": loss_threshold,
            "learning_rate": current_lr,
            "accumulator": dataset.accumulator,

        }, new_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break
    
    print(f"Saving the best model with validation loss {best_val_loss}")
    checkpoint = load_checkpoint(new_path)
    model.load_state_dict(checkpoint['state_dict'])
    return loss_list, val_loss_list, model