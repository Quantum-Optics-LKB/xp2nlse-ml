#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import argparse

import numpy as np

from engine.generate_data_for_training import generate_data
from engine.finder import lauch_training
from engine.use import get_parameters

parser = argparse.ArgumentParser(description='')

parser.add_argument('--device', type=int, default=0,
                    help='Which GPU you are using')

parser.add_argument('--saving_path', type=str, default="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN",
                    help='Directory path for saving output files.')
parser.add_argument('--input_image_path', type=str, default="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/input_beam.npy",
                    help='Path to the input image file. Default is /home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/input_beam.npy')
parser.add_argument('--input_power_path', type=str, default="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/laser_settings.npy",
                    help='Path to the powers. Default is /home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/laser_settings.npy')
parser.add_argument('--input_alpha_path', type=str, default="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/alpha.npy",
                    help='Path to the alpha. Default is /home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/alpha.npy')
parser.add_argument('--exp_image_path', type=str, default="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat/field.npy",
                    help='Path to the experiment image file. Default is /home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat/field.npy')

parser.add_argument('--resolution_in', type=int, default=1024,
                    help='Input resolution.')
parser.add_argument('--resolution_out', type=int, default=256,
                    help='Output resolution.')
parser.add_argument('--pin_size', type=float, default=2,
                    help='Size of pinhole')

parser.add_argument('--number_of_n2', type=int, default=10,
                    help='Number of different n2')
parser.add_argument('--number_of_isat', type=int, default=10,
                    help='Number of different Isat')

parser.add_argument('--expansion', action='store_true',
                    help='Enable expansion.')
parser.add_argument('--generate', action='store_true',
                    help='Enable generation.')

parser.add_argument('--delta_z', type=float, default=1e-4,
                    help='Step of the propagation of NLSE')
parser.add_argument('--length', type=float, default=20e-2,
                    help='Length of the cell')

parser.add_argument('--training', action='store_true',
                    help='Enable training.')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size')
parser.add_argument('--accumulator', type=int, default=2,
                    help='Number of accumulation steps to allow for gradient accumulation')
parser.add_argument('--num_epochs', type=int, default=60,
                    help='Number of epochs')

parser.add_argument('--use', action='store_true',
                    help='Find your parameters')

# Parse the arguments
args = parser.parse_args()

# Set the default for image_path if not specified

powers = np.load(args.input_power_path)
alpha = np.load(args.input_alpha_path)

power_alpha = powers, alpha

resolutions = args.resolution_in, args.resolution_out
numbers = args.number_of_n2, power_alpha, args.number_of_isat

labels, values = generate_data(args.saving_path, args.input_image_path, resolutions, numbers, 
                                args.generate, args.expansion, args.delta_z, args.length, 
                                         args.device, args.pin_size)

if args.training:
    print("---- TRAINING ----")
    lauch_training(numbers, labels, values, args.saving_path, args.resolution_out, args.learning_rate, args.batch_size, args.num_epochs, args.accumulator, args.device)

if args.use:
    print("---- COMPUTING PARAMETERS ----\n")
    get_parameters(args.exp_image_path, args.saving_path, args.resolution_out, numbers, args.device)