#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import argparse

from engine.generate_data_for_training import generate_data
from engine.finder import lauch_training
from engine.use import get_parameters

parser = argparse.ArgumentParser(description='')

parser.add_argument('--device', type=int, default=0,
                    help='Which GPU you are using')

parser.add_argument('--saving_path', type=str, default="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN",
                    help='Directory path for saving output files.')
parser.add_argument('--input_image_path', type=str, default=None,
                    help='Path to the input image file. Default is <saving_path>/exp_data/input_beam.tiff')
parser.add_argument('--exp_image_path', type=str, default=None,
                    help='Path to the experiment image file. Default is <saving_path>/exp_data/field.npy')
parser.add_argument('--output_image_path', type=str, default=None,
                    help='Path to the input image file. Default is <saving_path>/exp_data/input_beam.tiff')

parser.add_argument('--resolution_in', type=int, default=512,
                    help='Input resolution.')
parser.add_argument('--resolution_out', type=int, default=256,
                    help='Output resolution.')

parser.add_argument('--number_of_n2', type=int, default=10,
                    help='Number of different n2')
parser.add_argument('--number_of_power', type=int, default=10,
                    help='Number of different power')
parser.add_argument('--number_of_isat', type=int, default=10,
                    help='Number of different Isat')

parser.add_argument('--expansion', action='store_true',
                    help='Enable expansion.')
parser.add_argument('--generate', action='store_true',
                    help='Enable generation.')
parser.add_argument('--expanded', action='store_true',
                    help='Add if your data was expanded in a previous run')

parser.add_argument('--delta_z', type=float, default=1e-4,
                    help='Step of the propagation of NLSE')
parser.add_argument('--trans', type=float, default=0.01,
                    help='Transmission through the cell')
parser.add_argument('--length', type=float, default=20e-2,
                    help='Length of the cell')
parser.add_argument('--factor_window', type=int, default=56,
                    help='Factor window that is multiplied by the waist')

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
if args.input_image_path is None:
    args.input_image_path = f'{args.saving_path}/exp_data/input_beam.tiff'

if args.exp_image_path is None:
    args.exp_image_path = f'{args.saving_path}/exp_data/field.npy'

# You can now use args to access the values of the arguments
resolutions = args.resolution_in, args.resolution_out
numbers = args.number_of_n2, args.number_of_power, args.number_of_isat

labels, values = generate_data(args.saving_path, args.input_image_path, resolutions, numbers, 
                                args.generate,args.expanded, args.expansion, args.factor_window, args.delta_z, args.length, 
                                        args.trans, args.device)

if args.training:
    print("---- TRAINING ----")
    lauch_training(numbers, labels, values, args.saving_path, args.resolution_out, args.learning_rate, args.batch_size, args.num_epochs, args.accumulator, args.device)

if args.use:
    print("---- COMPUTING PARAMETERS ----\n")
    get_parameters(args.exp_image_path, args.saving_path, args.resolution_out, numbers, args.device)