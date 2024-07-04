#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from engine.nlse_sandbox import sandbox

saving_path="data"
device = 0

###Data generation Parameters:
delta_z=1e-4 #m
resolution_input_beam = 512
window_input = 20e-3 #m
output_camera_resolution = 2056
output_pixel_size = 3.45e-6 #m
window_out = output_pixel_size * output_camera_resolution #m
cell_length=20e-2 #m
resolution_training = 256

###Parameter spaces:
n2 = -4.24e-09 #switch this to an actual range using numpy to launch the real simulation 
isat = .6e4 #switch this to an actual range using numpy to launch the real simulation
alpha = 31.8 #switch this to an actual range using numpy to launch the real simulation

###Laser Parameters:
input_power = 0.570 #W
waist_input_beam = 3.564e-3 #m
non_locality_length = 0 #m

###Find your parameters (n2 and Isat):
exp_image_path="data/field.npy"

sandbox(device, resolution_input_beam, window_input, window_out,
        resolution_training, n2, input_power, alpha,
        isat, waist_input_beam, non_locality_length, delta_z,
        cell_length, exp_image_path, saving_path)