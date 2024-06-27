#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from engine.nlse_sandbox import sandbox

saving_path="/your/saving/path/"
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
n2 = -5.76008677482605e-09 #switch this to an actual range using numpy to launch the real simulation 
isat = 135309.72599983215 #switch this to an actual range using numpy to launch the real simulation
alpha = 31.8 #switch this to an actual range using numpy to launch the real simulation

###Laser Parameters:
input_power = 1.05 #W
waist_input_beam = 2.3e-3 #m
non_locality_length = 0 #m

###Find your parameters (n2 and Isat):
exp_image_path="/your/experiment/path/experiment.npy"

sandbox(device, resolution_input_beam, window_input, window_out,
        resolution_training, n2, input_power, alpha,
        isat, waist_input_beam, non_locality_length, delta_z,
        cell_length, exp_image_path, saving_path)