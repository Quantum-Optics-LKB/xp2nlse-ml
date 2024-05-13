#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np

from engine.generate_data_for_training import generate_data
from engine.finder import lauch_training
from engine.use import get_parameters

saving_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN"

input_image_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191652_time_flat_real/input_beam.npy"
input_power_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191652_time_flat_real/laser_settings.npy"
input_alpha_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191652_time_flat_real/alpha.npy"
exp_image_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191652_time_flat/field.npy"

device = 0
resolution_in=1024
resolution_out=256
number_of_n2=10
number_of_isat=10
n2 = np.linspace(-1e-9, -1e-11, number_of_n2) #m/W^2
isat = np.linspace(1e4, 1e6, number_of_isat) #W/m^2

delta_z=1e-4
length=20e-2
pin_size=2

expansion=False
generate=False
training=True
learning_rate=0.1
batch_size=50
accumulator=2
num_epochs=60

use=True

powers = np.load(input_power_path)
alpha = np.load(input_alpha_path)

resolutions = resolution_in, resolution_out
numbers = n2, powers, alpha, isat

values, E = generate_data(saving_path, input_image_path, resolutions, numbers, 
                                generate, expansion,training, delta_z, length, 
                                         device, pin_size)

if training:
    print("---- TRAINING ----")
    lauch_training(numbers, values,E, saving_path, resolution_out, learning_rate, batch_size, num_epochs, accumulator, device)

if use:
    print("---- COMPUTING PARAMETERS ----\n")
    get_parameters(exp_image_path, saving_path, resolution_out, numbers, device)