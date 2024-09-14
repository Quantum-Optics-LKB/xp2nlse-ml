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
    """
    Perform data augmentation on the provided dataset by adding fringes at random rotations, to the images.

    Args:
        E (np.ndarray): A 4D numpy array containing the image data with dimensions 
            (number_of_images, channels, height, width).
        labels (tuple): A tuple containing the number of each type of label and the 
            labels themselves. Expected to be in the format:
            (number_of_n2, n2_labels, number_of_isat, isat_labels, number_of_alpha, alpha_labels).

    Returns:
        np.ndarray: A 4D numpy array containing the augmented image data with dimensions 
            (augmented_number_of_images, channels, height, width).
        tuple: A tuple containing the augmented labels in the same format as the input labels.

    Description:
        This function augments the input image dataset by generating new images through 
        the following methods:
        - Fringes perturbations with fixed line counts (50 and 100) at different noises and angles.

        The augmentation process involves creating copies of the original images and 
        applying the aforementioned transformations to generate additional augmented images.

    Steps:
        1. Extract the individual label sets and counts from the input labels.
        2. Generate random angles and noise levels for augmentation.
        3. Determine the total number of augmented images to be created.
        4. Shuffle the original dataset and labels to ensure randomness.
        5. Repeat the labels to match the size of the augmented dataset.
        6. Initialize an array to hold the augmented data.
        7. Iterate through each image in the dataset:
            - Add original images without modifications.
            - Apply fringes at different angles perturbations to create augmented images.
        8. Return the augmented dataset and corresponding labels.
    """
    
    number_of_n2, n2_labels, number_of_isat, isat_labels, number_of_alpha, alpha_labels = labels

    angles = np.random.uniform(0,180,0)
    noises = np.random.uniform(0.1,0.4,0)
    lines = []#[50, 100]
    originals = 1
    augmentation = len(lines) * len(noises) * len(angles) + originals

    indices = np.arange(len(n2_labels))
    np.random.shuffle(indices)

    n2_labels = n2_labels[indices]
    isat_labels = isat_labels[indices]
    alpha_labels = alpha_labels[indices]
    E = E[indices, :, :, :]

    n2_labels = np.repeat(n2_labels, augmentation)
    isat_labels = np.repeat(isat_labels, augmentation)
    alpha_labels = np.repeat(alpha_labels, augmentation)

    
    labels = (number_of_n2, n2_labels, number_of_isat, isat_labels, number_of_alpha, alpha_labels)
    
    augmented_data = np.zeros((augmentation*E.shape[0], E.shape[1], E.shape[2],E.shape[3]), dtype=np.float16)

    index = 0
    for image_index in tqdm(range(E.shape[0]),desc=f"EXPANSION", 
                                total=number_of_n2*number_of_isat*number_of_alpha, unit="frame"):
        for original in range(originals):
            augmented_data[index,0 ,:, :] = E[image_index,0,:,:]
            augmented_data[index,1 ,:, :] = E[image_index,1,:,:]
            augmented_data[index,2 ,:, :] = E[image_index,2,:,:]
            index += 1  
        for noise in noises:
            for angle in angles:
                for num_lines in lines:
                    augmented_data[index,0 ,:, :] = normalize_data(line_noise(E[image_index,0,:,:], num_lines, np.max(E[image_index,0,:,:])*noise,angle))
                    augmented_data[index,1 ,:, :] = E[image_index,1,:,:]
                    augmented_data[index,2 ,:, :] = E[image_index,2,:,:]
                    index += 1
              
    return augmented_data, labels