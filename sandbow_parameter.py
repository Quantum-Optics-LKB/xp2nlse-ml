#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from engine.nlse_sandbox import sandbox

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

###Parameter spaces:
number_of_n2 = 1
number_of_isat = 1
n2 = -5e-9 #switch this to an actual range using numpy to launch the real simulation 
isat = 1e5 #switch this to an actual range using numpy to launch the real simulation

###Laser Parameters:
input_power = 1.05 #W
alpha = 22 #m^-1
waist_input_beam = 2.3e-3 #m
non_locality_length = 0 #m

###Find your parameters (n2 and Isat):
exp_image_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp/experiment.npy"

sandbox(device, resolution_input_beam, window_input, window_out,
        resolution_training, n2, input_power, alpha,
        isat, waist_input_beam, non_locality_length, delta_z,
        cell_length, exp_image_path, saving_path)