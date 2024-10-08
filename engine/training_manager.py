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
from engine.model import network
from engine.engine_dataset import EngineDataset
from engine.network_dataset import NetworkDataset
from engine.utils import data_split, plot_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from engine.training import load_checkpoint, network_training
set_seed(10)

def prepare_training(
        dataset: EngineDataset,
        ) -> tuple:
    
    
    device = torch.device("cpu")#dataset.device_number)
    new_path = f"{dataset.saving_path}/training_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}"
    os.makedirs(new_path, exist_ok=True)

    print("---- MODEL INITIALIZING ----")
    model = network().double()
    weight_decay = 1e-6
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=dataset.learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience = 5,factor = 0.5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)
    model = model.to(device)
    
    print("---- DATA TREATMENT ----")
    indices = np.arange(len(dataset.n2_labels))
    training_indices, validation_indices, test_indices = data_split(indices, 0.5, 0.25, 0.25)

    training_field = dataset.field[training_indices,:,:,:]
    validation_field = dataset.field[validation_indices,:,:,:]
    test_field = dataset.field[test_indices,:,:,:]

    training_n2_labels = dataset.n2_labels[training_indices]
    training_isat_labels = dataset.isat_labels[training_indices]
    training_alpha_labels = dataset.alpha_labels[training_indices]

    validation_n2_labels = dataset.n2_labels[validation_indices]
    validation_isat_labels = dataset.isat_labels[validation_indices]
    validation_alpha_labels = dataset.alpha_labels[validation_indices]

    test_n2_labels = dataset.n2_labels[test_indices]
    test_isat_labels = dataset.isat_labels[test_indices]
    test_alpha_labels = dataset.alpha_labels[test_indices]

    dataset.mean_standard = np.mean(training_field, axis=(0, -1, -2), keepdims=True)
    dataset.std_standard = np.std(training_field, axis=(0, -1, -2), keepdims=True)
    
    dataset.n2_min_standard = np.min(training_n2_labels)
    dataset.n2_max_standard = np.max(training_n2_labels)

    dataset.alpha_min_standard = np.min(training_alpha_labels)
    dataset.alpha_max_standard = np.max(training_alpha_labels)

    dataset.isat_min_standard = np.min(training_isat_labels)
    dataset.isat_max_standard = np.max(training_isat_labels)

    training_n2_labels -= dataset.n2_min_standard
    training_n2_labels /= dataset.n2_max_standard - dataset.n2_min_standard

    training_isat_labels -= dataset.isat_min_standard
    training_isat_labels /= dataset.isat_max_standard - dataset.isat_min_standard

    training_alpha_labels -= dataset.alpha_min_standard
    training_alpha_labels /= dataset.alpha_max_standard - dataset.alpha_min_standard

    validation_n2_labels -= dataset.n2_min_standard
    validation_n2_labels /= dataset.n2_max_standard - dataset.n2_min_standard

    validation_isat_labels -= dataset.isat_min_standard
    validation_isat_labels /= dataset.isat_max_standard - dataset.isat_min_standard

    validation_alpha_labels -= dataset.alpha_min_standard
    validation_alpha_labels /= dataset.alpha_max_standard - dataset.alpha_min_standard

    test_n2_labels -= dataset.n2_min_standard
    test_n2_labels /= dataset.n2_max_standard - dataset.n2_min_standard

    test_isat_labels -= dataset.isat_min_standard
    test_isat_labels /= dataset.isat_max_standard - dataset.isat_min_standard

    test_alpha_labels -= dataset.alpha_min_standard
    test_alpha_labels /= dataset.alpha_max_standard - dataset.alpha_min_standard

    training_field -= dataset.mean_standard
    training_field /= dataset.std_standard

    validation_field -= dataset.mean_standard
    validation_field /= dataset.std_standard

    test_field -= dataset.mean_standard
    test_field /= dataset.std_standard

    training_set = NetworkDataset(set=training_field, 
                                  n2_labels=training_n2_labels, 
                                  isat_labels=training_isat_labels, 
                                  alpha_labels=training_alpha_labels)

    validation_set = NetworkDataset(set=validation_field, 
                                    n2_labels=validation_n2_labels, 
                                    isat_labels=validation_isat_labels, 
                                    alpha_labels=validation_alpha_labels)

    test_set = NetworkDataset(set=test_field, 
                              n2_labels=test_n2_labels, 
                              isat_labels=test_isat_labels, 
                              alpha_labels=test_alpha_labels)
    
    
    model_settings = model, optimizer, criterion, scheduler, device, new_path

    return training_set, validation_set, test_set, model_settings

def manage_training(
        dataset: EngineDataset,
        training_set: NetworkDataset,
        validation_set: NetworkDataset, 
        test_set: NetworkDataset, 
        model_settings: tuple
        ) -> None:

    model, optimizer, criterion, scheduler, device, new_path = model_settings

    orig_stdout = sys.stdout
    f = open(f'{new_path}/testing.txt', 'a')
    sys.stdout = f

    try:
        checkpoint = load_checkpoint(new_path)
        model.load_state_dict(checkpoint['state_dict'])
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
    model_settings = model, optimizer, criterion, scheduler, device, new_path, start_epoch
    loss_list, val_loss_list, model = network_training(model_settings, dataset, training_set, validation_set, test_set, loss_list, val_loss_list)
    
    print("---- MODEL SAVING ----")
    directory_path = f'{new_path}/training_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}/'
    directory_path += f'n2_net_w{dataset.resolution_training}_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{dataset.input_power:.2f}.pth'
    torch.save(model.state_dict(), directory_path)

    file_name = f"{new_path}/params.txt"
    with open(file_name, "a") as file:
        file.write(f"n2: {dataset.n2_values} \n")
        file.write(f"alpha: {dataset.alpha_values}\n")
        file.write(f"Isat: {dataset.isat_values}\n")
        file.write(f"num_of_n2: {dataset.number_of_n2}\n")
        file.write(f"num_of_isat: {dataset.number_of_isat}\n")
        file.write(f"num_of_alpha: {dataset.number_of_alpha}\n")
        file.write(f"in_power: {dataset.input_power}\n")
        file.write(f"delta_z: {dataset.delta_z}\n")
        file.write(f"cell_length: {dataset.length}\n")
        file.write(f"waist_input_beam: {dataset.waist} m\n")
        file.write(f"non_locality_length: {dataset.non_locality} m\n")
        file.write(f"num_epochs: {dataset.num_epochs}\n")
        file.write(f"resolution training: {dataset.resolution_training}\n")
        file.write(f"resolution simulation: {dataset.resolution_simulation}\n")
        file.write(f"window training: {dataset.window_training}\n")
        file.write(f"window simulation: {dataset.window_simulation}\n")
        file.write(f"accumulator: {dataset.accumulator}\n")
        file.write(f"batch size: {dataset.batch_size}\n")
        file.write(f"learning_rate: {dataset.learning_rate}\n")
        
    plot_loss(loss_list,val_loss_list, new_path, dataset.resolution_training, dataset.number_of_n2, dataset.number_of_isat, dataset.number_of_alpha)

    exam(model_settings, test_set, dataset)

    sys.stdout = orig_stdout
    f.close()   