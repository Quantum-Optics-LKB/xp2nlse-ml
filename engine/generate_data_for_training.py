#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
import cupy as cp
from engine.nlse_generator import data_creation, data_augmentation

def generate_data(
        saving_path: str,
        resolution_training: int,
        numbers: tuple, 
        generate: bool,
        expansion: bool,
        device: int, 
        cameras: tuple
        )-> tuple:
    
    n2, in_power, alpha, isat, waist, nl_length, delta_z, length = numbers

    number_of_n2 = len(n2)
    number_of_isat = len(isat)

    N2_single, ISAT_single = np.meshgrid(n2, isat) 

    n2_single = N2_single.reshape(-1)
    isat_single = ISAT_single.reshape(-1)
    
    
    if generate:
        with cp.cuda.Device(device):
            print("---- NLSE ----")
            E = data_creation(numbers, cameras ,saving_path)
    else:
        file = f'{saving_path}/Es_w{resolution_training}_n2{number_of_n2}_isat{number_of_isat}_power{in_power:.2f}.npy'
        E = np.load(file)

    if expansion:

        print("---- EXPANSION ----")
        noise = 0.01
        expansion_factor, E = data_augmentation(number_of_n2, number_of_isat, in_power, E, noise, saving_path)

        n2_augmented_single = np.repeat(n2_single, expansion_factor)
        isat_augmented_single = np.repeat(isat_single, expansion_factor)

        values_augmented_single = (n2_augmented_single, isat_augmented_single)
        
        return values_augmented_single, E
    else:

        values_augmented_single = (n2_single, isat_single)
        return values_augmented_single, E