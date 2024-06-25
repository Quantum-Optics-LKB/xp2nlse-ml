#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from tqdm import tqdm
from engine.seed_settings import set_seed
from engine.treament_methods import normalize_data
from engine.treament_methods import line_noise, salt_and_pepper_noise
set_seed(10)

def data_augmentation(
    E: np.ndarray,
    labels: tuple,
    ) -> np.ndarray:
    
    number_of_n2, n2_labels, number_of_isat, isat_labels = labels

    angles = np.random.uniform(0,180,5)
    noises = np.random.uniform(0.1,0.4,2)
    lines = [20, 50, 100]
    augmentation = len(noises) + len(lines) * len(noises) * len(angles) + 1

    n2_labels = np.repeat(n2_labels, augmentation)
    isat_labels = np.repeat(isat_labels, augmentation)

    labels = (number_of_n2, n2_labels, number_of_isat, isat_labels)
    
    augmented_data = np.zeros((augmentation*E.shape[0], E.shape[1], E.shape[2],E.shape[3]), dtype=np.float16)

    index = 0
    for image_index in tqdm(range(E.shape[0]),desc=f"EXPANSION", 
                                total=number_of_n2*number_of_isat, unit="frame"):
        image_at_channel = E[image_index,0,:,:]
        augmented_data[index,0 ,:, :] = E[image_index,0,:,:]
        augmented_data[index,1 ,:, :] = E[image_index,1,:,:]
        augmented_data[index,2 ,:, :] = E[image_index,2,:,:]
        index += 1  
        for noise in noises:
            augmented_data[index,0 ,:, :] = normalize_data(salt_and_pepper_noise(image_at_channel, noise))
            augmented_data[index,1 ,:, :] = E[image_index,1,:,:]
            augmented_data[index,2 ,:, :] = E[image_index,1,:,:]

            index += 1
            for angle in angles:
                for num_lines in lines:
                    augmented_data[index,0 ,:, :] = normalize_data(line_noise(image_at_channel, num_lines, np.max(image_at_channel)*noise,angle))
                    augmented_data[index,1 ,:, :] = E[image_index,1,:,:]
                    augmented_data[index,2 ,:, :] = E[image_index,2,:,:]
                    index += 1
                    
    return augmented_data, labels