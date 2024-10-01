#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import os
import sys
import torch
import numpy as np
import torch.nn as nn
from engine.test import exam
from engine.utils import set_seed
from engine.model import Inception_ResNetv2
from engine.field_dataset import FieldDataset
from engine.utils import data_split, plot_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from engine.training import load_checkpoint, network_training
set_seed(10)

def network_init(
        learning_rate: float, 
        channels: int, 
        model: torch.nn.Module,
        num_epochs: int
        ) -> tuple:
    """
    Initializes the neural network model, criterion, optimizer, and scheduler.

    Args:
        learning_rate (float): The initial learning rate for the optimizer.
        channels (int): The number of channels in the input data.
        model (torch.nn.Module class): The neural network model class to be initialized.

    Returns:
        tuple: A tuple containing the initialized neural network model, optimizer, criterion, and scheduler.

    Description:
        This function initializes the neural network model (specified by `model`), creates an Adam optimizer with the specified learning rate, initializes Mean Squared Error (MSE) loss criterion, and sets up a ReduceLROnPlateau scheduler for the optimizer. It returns these initialized components as a tuple.
    """
    cnn = model(in_channels=channels).double()
    weight_decay = 1e-6
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience =5,factor=0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)


    return cnn, optimizer, criterion, scheduler

def prepare_training(
        nlse_settings: tuple, 
        labels: tuple, 
        E: np.ndarray,
        path: str, 
        learning_rate: float, 
        batch_size: int, 
        num_epochs: int, 
        accumulation_steps: int,
        device_number: torch.device
        ) -> tuple:

    """
    Prepares the training process by splitting data, initializing the model, and creating data loaders.

    Args:
        nlse_settings (tuple): Tuple containing NLSE settings (n2, input_power, alpha, isat, waist_input_beam, non_locality_length, delta_z, cell_length).
        labels (tuple): Tuple containing labels (number_of_n2, n2_values, number_of_isat, isat_values, number_of_alpha, alpha_values).
        E (np.ndarray): Input data array of shape (samples, channels, height, width).
        path (str): Path for saving training outputs.
        learning_rate (float): Initial learning rate for the optimizer.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs for training.
        accumulation_steps (int): Number of steps to accumulate gradients before optimizing.
        device_number (torch.device): Device number for training (e.g., GPU device).

    Returns:
        tuple: A tuple containing trainloader, validationloader, testloader, model_settings, and new_path.

    Description:
        This function initializes the model using `network_init`, splits the data into training, validation, and test sets, creates DataLoader objects for each set, and prepares model settings for training. It returns these components as a tuple.
    """
    
    device = torch.device("cpu")
    _, input_power, _, _, _, _, _, _ = nlse_settings

    number_of_n2, n2_values, number_of_isat, isat_values, number_of_alpha, alpha_values = labels
    
    n2_values_normalized = (n2_values - np.min(n2_values))/(np.max(n2_values) - np.min(n2_values))
    isat_values_normalized = (isat_values - np.min(isat_values))/(np.max(isat_values) - np.min(isat_values))
    alpha_values_normalized =(alpha_values - np.min(alpha_values))/(np.max(alpha_values) - np.min(alpha_values))

    new_path = f"{path}/training_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{input_power:.2f}"

    os.makedirs(new_path, exist_ok=True)

    assert E.shape[0] == n2_values.shape[0], f"field[0] is {E.shape[0]}, n2_values[0] is {n2_values.shape[0]}"
    assert E.shape[0] == isat_values.shape[0], f"field[0] is {E.shape[0]}, isat_values[0] is {isat_values.shape[0]}"
    assert E.shape[0] == alpha_values.shape[0], f"field[0] is {E.shape[0]}, alpha_values[0] is {alpha_values.shape[0]}"

    print("---- MODEL INITIALIZING ----")
    cnn, optimizer, criterion, scheduler = network_init(learning_rate, E.shape[1], Inception_ResNetv2, num_epochs=num_epochs)
    cnn = cnn.to(device)
    
    print("---- DATA TREATMENT ----")
    indices = np.arange(len(n2_values))
    np.random.shuffle(indices)

    training_indices, validation_indices, test_indices = data_split(indices, 0.9, 0.05, 0.05)
    fieldset = FieldDataset(data = E, training_indices=training_indices, validation_indices=validation_indices, test_indices=test_indices, 
                            batch_size=batch_size, n2_values=n2_values_normalized, isat_values=isat_values_normalized, alpha_values=alpha_values_normalized)
    model_settings = cnn, optimizer, criterion, scheduler, num_epochs, accumulation_steps, device

    return fieldset, model_settings, new_path

def manage_training(
        fieldset: FieldDataset,
        model_settings: tuple, 
        nlse_settings: tuple, 
        new_path: str, 
        resolution_training: int,
        labels: tuple
        ) -> None:
    """
    Manages the training process, including model initialization, training loop, and evaluation.

    Args:
        trainloader (DataLoader): DataLoader for training dataset.
        validationloader (DataLoader): DataLoader for validation dataset.
        testloader (DataLoader): DataLoader for test dataset.
        model_settings (tuple): Tuple containing model, optimizer, criterion, scheduler, num_epochs, accumulation_steps, and device.
        nlse_settings (tuple): Tuple containing NLSE settings (n2, input_power, alpha, isat, waist_input_beam, non_locality_length, delta_z, cell_length).
        new_path (str): Path for saving training outputs.
        resolution_training (int): Resolution of the training data.
        labels (tuple): Tuple containing labels (number_of_n2, n2_values, number_of_isat, isat_values, number_of_alpha, alpha_values).

    Returns:
        None

    Description:
        This function manages the entire training process, including model initialization, training loop execution using `network_training`, model saving, and evaluation using `exam`. It also saves training parameters and plots loss curves.
    """

    number_of_n2, _, number_of_isat, _, number_of_alpha, _ = labels
    n2, input_power, alpha, isat, waist_input_beam, non_locality_length, _, _ = nlse_settings
    cnn, optimizer, criterion, scheduler, num_epochs, accumulation_steps, device = model_settings

    orig_stdout = sys.stdout
    f = open(f'{new_path}/testing.txt', 'a')
    sys.stdout = f

    try:
        checkpoint = load_checkpoint(new_path)
        cnn.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        loss_list = checkpoint['loss_list']
        val_loss_list = checkpoint['val_loss_list']
    except FileNotFoundError:
        start_epoch = 0
        loss_list = []
        val_loss_list = []
    
    print("---- MODEL TRAINING ----")
    loss_list, val_loss_list, cnn = network_training(cnn, optimizer, criterion, scheduler,start_epoch, num_epochs, fieldset, accumulation_steps, device, new_path, loss_list, val_loss_list)
    
    print("---- MODEL SAVING ----")
    directory_path = f'{new_path}/training_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{input_power:.2f}/'
    directory_path += f'n2_net_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{input_power:.2f}.pth'
    torch.save(cnn.state_dict(), f'{new_path}/n2_net_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{input_power:.2f}.pth')

    file_name = f"{new_path}/params.txt"
    with open(file_name, "a") as file:
        file.write(f"n2: {n2} \n")
        file.write(f"alpha: {alpha}\n")
        file.write(f"Isat: {isat}\n")
        file.write(f"num_of_n2: {number_of_n2}\n")
        file.write(f"num_of_isat: {number_of_isat}\n")
        file.write(f"num_of_alpha: {number_of_alpha}\n")
        file.write(f"in_power: {input_power}\n")
        file.write(f"waist_input_beam: {waist_input_beam} m\n")
        file.write(f"non_locality_length: {non_locality_length} m\n")
        file.write(f"num_epochs: {num_epochs}\n")
        file.write(f"model: {Inception_ResNetv2}\n")
        file.write(f"resolution: {resolution_training}\n")
        file.write(f"accumulator: {accumulation_steps}\n")
        
    plot_loss(loss_list,val_loss_list, new_path, resolution_training,number_of_n2,number_of_isat, number_of_alpha)

    exam(cnn, fieldset, device, new_path)

    sys.stdout = orig_stdout
    f.close()   