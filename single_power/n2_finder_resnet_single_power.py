#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import os
import sys
from tqdm import tqdm
import torch
import numpy as np
from loss_plot_2D import plotter
from single_power.n2_test_resnet_single_power import count_parameters_pandas, test_model_classification
from data_prep_for_training import data_split, data_treatment
from single_power.n2_training_resnet_single_power import network_training
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

def network_init(
        learning_rate: float, 
        channels: int, 
        class_n2: int, 
        class_power: int, 
        class_isat: int, 
        model: torch.nn.Module
        ) -> tuple:
    """
    Initializes the neural network model, criterion, optimizer, and scheduler.

    Parameters:
    - learning_rate (float): The initial learning rate for the optimizer.
    - channels (int): The number of channels in the input data.
    - class_n2 (int): The number of classes for n2 predictions.
    - class_power (int): The number of classes for power predictions.
    - class_isat (int): The number of classes for isat predictions.
    - model (torch.nn.Module class): The neural network model class to be initialized.

    Returns:
    - cnn (torch.nn.Module): The initialized neural network model.
    - optimizer (torch.optim.Optimizer): The Adam optimizer initialized with the model parameters.
    - criterion (torch.nn.Module): The CrossEntropyLoss criterion for the model's output.
    - scheduler (torch.optim.lr_scheduler): A ReduceLROnPlateau learning rate scheduler.
    """
    cnn = model(channels, class_isat, class_n2, class_power)
    weight_decay = 1e-5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    return cnn, optimizer, criterion, scheduler

def lauch_training(
        numbers: tuple, 
        labels: tuple, 
        values: tuple, 
        path: str, 
        resolution: int, 
        learning_rate: float, 
        batch_size: int, 
        num_epochs: int, 
        accumulation_steps: int
        ) -> None:
    """
    Prepares the dataset and launches the training process for a neural network model.

    Parameters:
    - numbers (tuple): A tuple containing the numbers of n2, power, and isat instances.
    - labels (tuple): A tuple containing n2 labels, power labels, and isat labels arrays.
    - values (tuple): A tuple containing n2 values, power values, and isat values arrays.
    - path (str): The path where training outputs will be saved.
    - resolution (int): The resolution of the input data.
    - learning_rate (float): The learning rate for the optimizer.
    - batch_size (int): The size of batches for training.
    - num_epochs (int): The number of training epochs.
    - accumulation_steps (int): The number of steps to accumulate gradients before an optimizer step.

    This function iterates over different power values and model configurations to train multiple models. 
    For each model and power configuration, it initializes the model, splits the dataset, prepares DataLoader
    objects, and then trains the model using the `network_training` function. Results, including losses and 
    the trained model, are saved to the specified path.
    """
    number_of_n2, number_of_power, number_of_isat = numbers
    n2_labels, power_labels, isat_labels = labels
    n2_values, power_values, isat_values = values
    backend = "GPU"
    device = torch.device("cuda:0")

    data_types = ["amp", "amp_pha", "amp_pha_unwrap", "pha", "pha_unwrap", "amp_pha_pha_unwrap"]

    power_index = 0
    powers = np.linspace(np.min(power_values), np.max(power_values), number_of_power)
    n2 = np.linspace(np.max(n2_values), np.min(n2_values), number_of_n2)
    isat = np.linspace(np.min(isat_values), np.max(isat_values), number_of_isat)
    
    for power in tqdm(powers, position=4,desc="Iteration", leave=False):

        power_index += 1
        for model_index in range(2, 6):
            if model_index == 2:
                from single_power.model.model_resnetv2_1powers import Inception_ResNetv2
            elif model_index == 3:
                from single_power.model.model_resnetv3_1powers import Inception_ResNetv2
            elif model_index == 4:
                from single_power.model.model_resnetv4_1powers import Inception_ResNetv2
            elif model_index == 5:
                from single_power.model.model_resnetv5_1powers import Inception_ResNetv2

            for data_types_index in range(len(data_types)):
            
                model_version =  str(Inception_ResNetv2).split('.')[0][8:]
                stamp = f"power{str(power)[:4]}_{data_types[data_types_index]}_{model_version}"
                
                if not os.path.isdir(f"{path}/{stamp}_training"):
                    os.makedirs(f"{path}/{stamp}_training")
                new_path = f"{path}/{stamp}_training"

                orig_stdout = sys.stdout
                f = open(f'{new_path}/testing.txt', 'a')
                sys.stdout = f

                print("---- DATA LOADING ----")

                # number_of_n2 x 1 x number_of_isat
                file = f'{path}/Es_w{resolution}_n2{number_of_n2}_isat{number_of_isat}_power{1}_at{str(power)[:4]}_amp_pha_pha_unwrap_extended.npy' 
                
                
                if data_types_index == 0:
                    E_noisy = np.load(file, 'r')[:,[0],:,:]
                    assert E_noisy.shape[1] == 1
                elif data_types_index == 1:
                    E_noisy = np.load(file, 'r')[:,[0,1],:,:]
                    assert E_noisy.shape[1] == 2
                elif data_types_index == 2:
                    E_noisy = np.load(file, 'r')[:,[0,2],:,:]
                    assert E_noisy.shape[1] == 2
                elif data_types_index == 3:
                    E_noisy = np.load(file, 'r')[:,[1],:,:]
                    assert E_noisy.shape[1] == 1
                elif data_types_index == 4:
                    E_noisy = np.load(file, 'r')[:,[2],:,:]
                    assert E_noisy.shape[1] == 1
                elif data_types_index == 5:
                    E_noisy = np.load(file, 'r')
                    assert E_noisy.shape[1] == 3


                assert E_noisy.shape[0] == n2_labels.shape[0]
                assert E_noisy.shape[0] == power_labels.shape[0]
                assert E_noisy.shape[0] == isat_labels.shape[0]
                assert E_noisy.shape[0] == n2_values.shape[0]
                assert E_noisy.shape[0] == power_values.shape[0]
                assert E_noisy.shape[0] == isat_values.shape[0]

                
                print("---- MODEL INITIALIZING ----")
                cnn, optimizer, criterion, scheduler = network_init(learning_rate, E_noisy.shape[1], number_of_n2,number_of_power,number_of_isat, Inception_ResNetv2)
                cnn = cnn.to(device)
                
                print("---- DATA TREATMENT ----")
                train_set, validation_set, test_set = data_split(E_noisy,n2_labels, power_labels,isat_labels, power_values, 0.8, 0.1, 0.1)

                train, train_n2_label, train_power_label,train_isat_label, train_power_value = train_set
                validation, validation_n2_label, validation_power_label, validation_isat_label, validation_power_value = validation_set
                test, test_n2_label, test_power_label, test_isat_label, test_power_value = test_set

                training_train = True
                training_valid = False
                training_test = False

                trainloader = data_treatment(train, train_n2_label, train_power_label,train_isat_label, train_power_value, batch_size, device, training_train)
                validationloader = data_treatment(validation, validation_n2_label, validation_power_label, validation_isat_label, validation_power_value, batch_size, device, training_valid)
                testloader = data_treatment(test, test_n2_label, test_power_label, test_isat_label, test_power_value, batch_size, device, training_test )

                print("---- MODEL TRAINING ----")
                loss_list, val_loss_list, cnn = network_training(cnn, optimizer, criterion, scheduler, num_epochs, trainloader, validationloader, device, accumulation_steps, backend)

                print("---- MODEL SAVING ----")
                torch.save(cnn.state_dict(), f'{new_path}/n2_net_w{resolution}_n2{number_of_n2}_isat{number_of_isat}_power{1}.pth')

                file_name = f"{new_path}/params.txt"
                classes = {
                    'n2': tuple(map(str, n2)),
                    'isat' : tuple(map(str, isat))
                }
                with open(file_name, "a") as file:
                    file.write(f"power: {power}\n")
                    file.write(f"resolution: {resolution}\n")
                    file.write(f"batch_size: {batch_size}\n")
                    file.write(f"accumulator: {accumulation_steps}\n")
                    file.write(f"num_of_n2: {number_of_n2}\n")
                    file.write(f"num_of_power: {1}\n")
                    file.write(f"num_epochs: {num_epochs}\n")
                    file.write(f"learning rate: {learning_rate}\n")
                    file.write(f"file: {file}\n")
                    file.write(f"training_train: {training_train}\n")
                    file.write(f"training_valid: {training_valid}\n")
                    file.write(f"training_test: {training_test}\n")
                    file.write(f"model: {Inception_ResNetv2}\n")
                    file.write(f"classes: {classes}\n")

                plotter(loss_list,val_loss_list, new_path, resolution,number_of_n2,number_of_power)

                print("---- MODEL ANALYSIS ----")
                count_parameters_pandas(cnn)

                print("---- MODEL TESTING ----")
                test_model_classification(testloader, cnn, classes, device, backend)

                sys.stdout = orig_stdout
                f.close()