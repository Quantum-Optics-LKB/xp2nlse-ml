#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from .seed_settings import set_seed
set_seed(42)

def manager(generate: bool, 
            training: bool, 
            create_visual: bool,
            use: bool,
            plot_generate_compare: bool,
            device: int, 
            resolution_input_beam: int,
            window_input: float,
            window_out: float,
            resolution_training: int,
            n2: np.ndarray,
            number_of_n2: int,
            input_power: float,
            alpha: float,
            isat: float,
            number_of_isat: int,
            waist_input_beam: float,
            non_locality_length: float,
            delta_z: float,
            cell_length: float, 
            saving_path: str,
            exp_image_path: str,
            learning_rate: float, 
            batch_size: int, 
            num_epochs: int, 
            accumulator: int
            ):
    
    cameras = resolution_input_beam, window_input, window_out, resolution_training
    nlse_settings = n2, input_power, alpha, isat, waist_input_beam, non_locality_length, delta_z, cell_length

    if generate or training:
        from engine.generate_augment import data_creation, data_augmentation, generate_labels
        if generate:
            import cupy as cp
            with cp.cuda.Device(device):
                E = data_creation(nlse_settings, cameras, saving_path)
        else:
            E = np.load(f'{saving_path}/Es_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_power{input_power:.2f}.npy')
        if create_visual:
            from engine.visualize import plot_and_save_images
            plot_and_save_images(E, saving_path, nlse_settings)

        labels = generate_labels(n2, isat)
        E, labels = data_augmentation(E, labels)

    if training:
        from engine.finder import launch_training, prep_training
        import gc
        print("---- TRAINING ----")
        trainloader, validationloader, testloader, model_settings, new_path = prep_training(nlse_settings, labels, E, saving_path, learning_rate, batch_size, num_epochs, accumulator, device)
        del E
        gc.collect()
        launch_training(trainloader, validationloader, testloader, model_settings, nlse_settings, new_path, resolution_training, labels)

    if use:
        from engine.use import get_parameters
        print("---- COMPUTING PARAMETERS ----\n")
        get_parameters(exp_image_path, saving_path, resolution_training, nlse_settings, device, cameras, plot_generate_compare)