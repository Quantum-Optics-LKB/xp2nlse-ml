#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import cupy as cp
import numpy as np
from engine.use import get_parameters
from engine.generate import simulation
from engine.engine_dataset import EngineDataset
from engine.utils import plot_generated_set, set_seed
from engine.training_manager import manage_training, prepare_training
set_seed(10)

def manager(
        generate: bool, 
        training: bool, 
        create_visual: bool,
        use: bool,
        plot_generate_compare: bool,
        window_training: float,
        n2_values: np.ndarray,
        alpha_values: float,
        isat_values: float,
        input_power: float,
        waist: float,
        length: float, 
        saving_path: str,
        exp_image_path: str,
        device_number: int = 0, 
        resolution_simulation: int = 512,
        window_simulation: float = 20e-3,
        resolution_training: int = 256,
        non_locality: float = 0,
        delta_z: float = 1e-4,
        learning_rate: float = 0.01, 
        batch_size: int = 100, 
        num_epochs: int = 30, 
        accumulator: int = 1
        ) -> None:
    
    dataset = EngineDataset(n2_values=n2_values, alpha_values=alpha_values, isat_values=isat_values, 
                           input_power=input_power, waist=waist, non_locality=non_locality, delta_z=delta_z,
                           length=length, resolution_simulation=resolution_simulation, resolution_training=resolution_training, 
                           window_simulation=window_simulation, window_training=window_training, saving_path=saving_path, learning_rate=learning_rate, 
                           batch_size=batch_size, num_epochs=num_epochs, accumulator=accumulator, device_number=device_number)

    if generate or training:
        if generate:
            with cp.cuda.Device(device_number):
                simulation(dataset)
        else:
            path = f'{saving_path}/Es_w{resolution_training}_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{input_power:.2f}.npy'
            dataset.field = np.load(path)
        
        min_values = np.min(dataset.field[:,0,:,:], axis=(-2, -1), keepdims=True)
        np.subtract(dataset.field[:,0,:,:], min_values, out=dataset.field[:,0,:,:])

        max_values = np.max(dataset.field[:,0,:,:], axis=(-2, -1), keepdims=True)
        np.divide(dataset.field[:,0,:,:], max_values, out=dataset.field[:,0,:,:])

        dataset.field[:, 1, :, :] = (dataset.field[:, 1, :, :] + np.pi) / (2 * np.pi)

        if create_visual:
            plot_generated_set(dataset)

        if training:
            print("---- TRAINING ----")
            training_set, validation_set, test_set, model_settings = prepare_training(dataset)
            manage_training(dataset, training_set, validation_set, test_set, model_settings)
    
    if not generate and not training and create_visual:
        path = f'{saving_path}/Es_w{resolution_training}_n2{dataset.number_of_n2}_isat{dataset.number_of_isat}_alpha{dataset.number_of_alpha}_power{input_power:.2f}.npy'
        dataset.field = np.load(path)

        min_values = np.min(dataset.field[:,0,:,:], axis=(-2, -1), keepdims=True)
        np.subtract(dataset.field[:,0,:,:], min_values, out=dataset.field[:,0,:,:])

        max_values = np.max(dataset.field[:,0,:,:], axis=(-2, -1), keepdims=True)
        np.divide(dataset.field[:,0,:,:], max_values, out=dataset.field[:,0,:,:])

        dataset.field[:, 1, :, :] = (dataset.field[:, 1, :, :] + np.pi) / (2 * np.pi)
        plot_generated_set(dataset)


    if use:
        print("---- COMPUTING PARAMETERS ----\n")
        get_parameters(exp_image_path, dataset, plot_generate_compare)