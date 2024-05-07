#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np

from engine.generate_data_for_training import generate_data
from engine.finder import lauch_training
from engine.use import get_parameters

saving_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN",

input_image_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/input_beam.npy"
input_power_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/laser_settings.npy"
input_alpha_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/alpha.npy"
exp_image_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat/field.npy"

device = 1
resolution_in=1024
resolution_out=256
pin_size=2
number_of_n2=10
number_of_isat=10
expansion=True
generate=True

delta_z=1e-4
length=20e-2
training=False
learning_rate=0.001
batch_size=50
accumulator=2
num_epochs=60

use=False

powers = np.load(input_power_path)
alpha = np.load(input_alpha_path)

power_alpha = powers, alpha

resolutions = resolution_in, resolution_out
numbers = number_of_n2, power_alpha, number_of_isat

labels, values, E = generate_data(saving_path, input_image_path, resolutions, numbers, 
                                generate, expansion,training, delta_z, length, 
                                         device, pin_size)

if training:
    print("---- TRAINING ----")
    lauch_training(numbers, labels, values,E, saving_path, resolution_out, learning_rate, batch_size, num_epochs, accumulator, device)

if use:
    print("---- COMPUTING PARAMETERS ----\n")
    get_parameters(exp_image_path, saving_path, resolution_out, numbers, device)