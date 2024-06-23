#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from engine.parameter_manager import manager
saving_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN"
device = 0

###Data generation Parameters:
delta_z=1e-4 #m
resolution_input_beam = 2048
window_input = 50e-3 #m
output_camera_resolution = 3008
output_pixel_size = 3.76e-6 #m
window_out = output_pixel_size * output_camera_resolution #m
cell_length=20e-2 #m
resolution_training = 256
generate=False
create_visual = False

###Parameter spaces:
number_of_n2 = 40
number_of_isat = 40
n2 = -5*np.logspace(-11, -9, number_of_n2) #m/W^2 [-5e-11 -> -5e-9]
isat = np.logspace(4, 5, number_of_isat) #W/m^2 [1e4 -> 1e5]

###Laser Parameters:
input_power = 1.05 #W
alpha = 22 #m^-1
waist_input_beam = 2.3e-3 #m
non_locality_length = 0 #m

###Training Parameters:
training=False
learning_rate=0.01
batch_size=100
accumulator=1
num_epochs=100

###Find your parameters (n2 and Isat):
exp_image_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp/experiment.npy"
use=False
plot_generate_compare=False

manager(generate, training, create_visual, use, plot_generate_compare, device, 
            resolution_input_beam, window_input, window_out, resolution_training, n2, number_of_n2,
            input_power, alpha, isat, number_of_isat, waist_input_beam, non_locality_length, delta_z, cell_length, 
            saving_path, exp_image_path, learning_rate, batch_size, num_epochs, accumulator)