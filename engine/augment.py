#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from tqdm import tqdm
from engine.utils import set_seed
from engine.utils import line_noise
from engine.utils import normalize_data
set_seed(10)

def data_augmentation(
    E: np.ndarray,
    labels: tuple,
    ) -> np.ndarray:
    
    number_of_n2, n2_labels, number_of_isat, isat_labels, number_of_alpha, alpha_labels = labels

    angles = np.random.uniform(0,180,4)
    noises = np.random.uniform(0.1,0.4,1)
    lines = [50, 100]
    originals = 16
    augmentation = len(lines) * len(noises) * len(angles) + originals

    n2_labels = np.repeat(n2_labels, augmentation)
    isat_labels = np.repeat(isat_labels, augmentation)
    alpha_labels = np.repeat(alpha_labels, augmentation)

    indices = np.arange(len(n2_labels))
    np.random.shuffle(indices)

    n2_labels = n2_labels[indices]
    isat_labels = isat_labels[indices]
    alpha_labels = alpha_labels[indices]

    labels = (number_of_n2, n2_labels, number_of_isat, isat_labels, number_of_alpha, alpha_labels)
    
    augmented_data = np.zeros((augmentation*E.shape[0], E.shape[1], E.shape[2],E.shape[3]), dtype=np.float16)

    index = 0
    for image_index in tqdm(range(E.shape[0]),desc=f"EXPANSION", 
                                total=number_of_n2*number_of_isat*number_of_alpha, unit="frame"):
        for original in range(originals):
            augmented_data[indices[index],0 ,:, :] = E[image_index,0,:,:]
            augmented_data[indices[index],1 ,:, :] = E[image_index,1,:,:]
            augmented_data[indices[index],2 ,:, :] = E[image_index,2,:,:]
            index += 1  
        for noise in noises:
            for angle in angles:
                for num_lines in lines:
                    augmented_data[indices[index],0 ,:, :] = normalize_data(line_noise(E[image_index,0,:,:], num_lines, np.max(E[image_index,0,:,:])*noise,angle))
                    augmented_data[indices[index],1 ,:, :] = E[image_index,1,:,:]
                    augmented_data[indices[index],2 ,:, :] = E[image_index,2,:,:]
                    index += 1
              
    return augmented_data, labels