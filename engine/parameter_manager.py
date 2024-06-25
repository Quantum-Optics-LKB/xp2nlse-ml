#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from .seed_settings import set_seed
set_seed(10)

def manager(generate: bool, 
            training: bool, 
            create_visual: bool,
            use: bool,
            plot_generate_compare: bool,
            window_out: float,
            n2: np.ndarray,
            number_of_n2: int,
            alpha: float,
            isat: float,
            number_of_isat: int,
            input_power: float,
            waist_input_beam: float,
            cell_length: float, 
            saving_path: str,
            exp_image_path: str,
            device: int = 0, 
            resolution_input_beam: int = 2048,
            window_input: float = 50e-3,
            resolution_training: int = 256,
            non_locality_length: float = 0,
            delta_z: float = 1e-4,
            learning_rate: float = 0.01, 
            batch_size: int = 100, 
            num_epochs: int = 60, 
            accumulator: int = 1
            ) -> None:
    
    cameras = resolution_input_beam, window_input, window_out, resolution_training
    nlse_settings = n2, input_power, alpha, isat, waist_input_beam, non_locality_length, delta_z, cell_length

    if generate or training:
        import gc
        import cupy as cp
        from engine.generate import data_creation, generate_labels
        from engine.augment import data_augmentation
        from engine.visualize import plot_and_save_images
        from engine.finder import manage_training, prepare_training
        
        
        if generate:
            with cp.cuda.Device(device):
                E = data_creation(nlse_settings, cameras, saving_path)
        else:
            E = np.load(f'{saving_path}/Es_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_power{input_power:.2f}.npy')
        if create_visual:
            
            plot_and_save_images(E, saving_path, nlse_settings)

        labels = generate_labels(n2, isat)
        E, labels = data_augmentation(E, labels)

        if training:
            print("---- TRAINING ----")
            trainloader, validationloader, testloader, model_settings, new_path = prepare_training(nlse_settings, labels, E, saving_path, 
                                                                                                   learning_rate, batch_size, num_epochs, 
                                                                                                    accumulator, device)
            del E
            gc.collect()
            manage_training(trainloader, validationloader, testloader, model_settings, 
                            nlse_settings, new_path, resolution_training, labels)

    if use:
        from engine.use import get_parameters
        print("---- COMPUTING PARAMETERS ----\n")
        get_parameters(exp_image_path, saving_path, resolution_training, nlse_settings, device, cameras, plot_generate_compare)