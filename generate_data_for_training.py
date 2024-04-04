#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
from nlse_generator import data_creation, data_augmentation
from PIL import Image
from waist_fitting import waist_computation
from scipy.ndimage import zoom
from matplotlib.colors import LogNorm


def from_input_image(
        path: str,
        number_of_power: int,
        number_of_n2: int,
        number_of_isat: int,
        resolution_in: int
        ) -> np.ndarray:
    """
    Loads an input image from the specified path and prepares it for nonlinear Schrödinger equation (NLSE) analysis by
    normalizing and tiling the image array based on specified parameters. The function also computes an approximation
    of the beam waist within the image.

    Parameters:
    - path (str): The file path to the input image. Currently supports TIFF format. Assumed to be square image
    - number_of_power (int): The number of different power levels for which the input field will be tiled.
    - number_of_n2 (int): The number of different nonlinear refractive index (n2) values for tiling.
    - number_of_isat (int): The number of different saturation intensities (Isat) for tiling.
    - resolution_in (float): The spatial resolution of the input image in meters per pixel.

    Returns:
    - input_field_tiled_n2_power_isat (np.ndarray): A 5D numpy array of the tiled input field adjusted for different
      n2, power, and Isat values. The array shape is (number_of_n2, number_of_power, number_of_isat, height, width),
      where height and width correspond to the dimensions of the input image.
    - waist (float): An approximation of the beam waist in meters, calculated based on the input image and resolution.
    """
    print("--- LOAD INPUT IMAGE ---")
    input_tiff = Image.open(path)
    image = np.array(input_tiff, dtype=np.float32)
    if resolution_in != image.shape[0]:
        image = zoom(image, (resolution_in/image.shape[0],resolution_in/image.shape[1]), order=5)
    print("--- FIND WAIST ---")
    window = resolution_in * 5.5e-6
    # waist = waist_computation(image, window, resolution_in, resolution_in, False)
    waist = 0.00043868048364296215

    print("--- PREPARE FOR NLSE ---")
    # array_normalized = (image - np.min(image) )/ (np.max(image) - np.min(image))
    input_field = image + 1j * np.zeros_like(image, dtype=np.float32)
    input_field_tiled_n2_power_isat = np.tile(input_field[np.newaxis,np.newaxis, np.newaxis, :,:], (number_of_n2,number_of_power,number_of_isat, 1,1))

    return input_field_tiled_n2_power_isat, waist

def from_gaussian(
        NX: int,
        NY: int, 
        window: int,
        waist: float 
        ) ->  np.ndarray:
    """
    Generates a 2D Gaussian distribution based on the specified dimensions, window size, and waist. This function is
    designed to create a Gaussian beam profile that can be used for simulations or analysis in optical physics and
    related fields.

    Parameters:
    - NX (int): The number of points in the X dimension.
    - NY (int): The number of points in the Y dimension.
    - window (float): The size of the square window in meters through which the Gaussian beam is defined. The function
      will generate points within this window centered around zero.
    - waist (float): The waist (radius at which the field amplitude falls to 1/e of its axial value) of the Gaussian
      beam in meters. This parameter determines the spread of the Gaussian distribution.

    Returns:
    - np.ndarray: A 2D numpy array representing the Gaussian beam intensity distribution over a meshgrid defined by
      the NX and NY points within the specified window. The intensity values are calculated using the formula
      exp(-(x^2 + y^2) / waist^2), where x and y are the coordinates of each point in the meshgrid.
    """
    X, delta_X = np.linspace(
        -window / 2,
        window / 2,
        num=NX,
        endpoint=False,
        retstep=True,
        dtype=np.float32,
    )
    Y, delta_Y = np.linspace(
        -window / 2,
        window / 2,
        num=NY,
        endpoint=False,
        retstep=True,
        dtype=np.float32,
    )

    XX, YY = np.meshgrid(X, Y)

    return np.ones((number_of_n2,number_of_power,number_of_isat, resolution_in, resolution_in), dtype=precision) * np.exp(-(XX**2 + YY**2) / (waist**2))

path = "/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN"
resolution_in = 512
resolution_out = 512

number_of_n2 = 10
number_of_power = 10
number_of_isat = 10

is_from_image = True
visualize = False
expension = True
generate = True
precision = np.complex64
delta_z = 1e-5
trans = 0.01
length = 20e-2
factor_window = 13

n2_values = np.linspace(-1e-11, -1e-10, number_of_n2)
n2_labels = np.arange(0, number_of_n2)

power_values = np.linspace(.02, 0.5001, number_of_power)
power_labels = np.arange(0, number_of_power)

isat_values = np.linspace(1e4, 1e6, number_of_isat)
isat_labels = np.arange(0, number_of_isat)

N2_values, POWER_values, ISAT_values = np.meshgrid(n2_values,power_values, isat_values,) 
N2_labels, POWER_labels, ISAT_labels = np.meshgrid(n2_labels, power_labels, isat_labels)

power_values_all = POWER_values.reshape((number_of_power*number_of_n2*number_of_isat,))
n2_values_all = N2_values.reshape((number_of_power*number_of_n2*number_of_isat,))
isat_values_all = ISAT_values.reshape((number_of_power*number_of_n2*number_of_isat,))

power_labels_all = POWER_labels.reshape((number_of_power*number_of_n2*number_of_isat,))
n2_labels_all = N2_labels.reshape((number_of_power*number_of_n2*number_of_isat,))
isat_labels_all = ISAT_labels.reshape((number_of_power*number_of_n2*number_of_isat,))

if is_from_image:
    input_field, waist = from_input_image(f'{path}/exp_data/input_beam.tiff', number_of_power, number_of_n2, number_of_isat, resolution_in)
    window = factor_window*waist
else:
    waist = 1e-3
    window = factor_window*waist
    input_field = from_gaussian(resolution_in, resolution_in, window, waist, precision)

if generate:
    with cp.cuda.Device(0):
        print("---- NLSE ----")
        E_clean = data_creation(input_field, window, n2_values,power_values,isat_values, resolution_in,resolution_out, delta_z,trans, length,path)
else:
    E_clean = np.load(f"{path}/Es_w{resolution_out}_n2{number_of_n2}_Isat{number_of_isat}_power{number_of_power}_amp_pha_pha_unwrap_all.npy")

if expension:
    noise = 0.01
    power = 0
    E_expend = data_augmentation(number_of_n2, number_of_power, number_of_isat, power, E_clean, noise, path)

if visualize:
    print("---- VISUALIZE ----")

    data_types = ["amp", "pha", "pha_unwrap"]
    cmap_types = ["viridis", "twilight_shifted", "viridis"]

    for data_types_index in range(len(data_types)):
        counter = 0
        for power_index in range(number_of_power):

            if number_of_isat > number_of_n2:
                fig, axs = plt.subplots(number_of_isat,number_of_n2, figsize=(number_of_n2*5, number_of_isat*5))
            else:
                fig, axs = plt.subplots(number_of_n2, number_of_isat, figsize=(number_of_n2*5, number_of_isat*5))
            
            for n2_index in range(number_of_n2):
                for isat_index in range(number_of_isat):
                    if number_of_isat == 1 and number_of_n2 == 1:
                        axs.imshow(E_clean[counter, data_types_index,:, :], cmap=cmap_types[data_types_index])
                        axs.set_title(f'power={power_values[power_labels_all[counter]]}, n2={n2_values[n2_labels_all[counter]]}, Isat={"{:e}".format(isat_values[isat_labels_all[counter]])}')
                        plt.axis('off')
                    elif number_of_isat == 1:
                        axs[n2_index].imshow(E_clean[counter, data_types_index,:, :], cmap=cmap_types[data_types_index])
                        axs[n2_index].set_title(f'power={power_values[power_labels_all[counter]]}, n2={n2_values[n2_labels_all[counter]]}, Isat={"{:e}".format(isat_values[isat_labels_all[counter]])}')
                        plt.axis('off')
                    elif number_of_n2 == 1:
                        axs[isat_index].imshow(E_clean[counter, data_types_index,:, :], cmap=cmap_types[data_types_index])
                        axs[isat_index].set_title(f'power={power_values[power_labels_all[counter]]}, n2={n2_values[n2_labels_all[counter]]}, Isat={"{:e}".format(isat_values[isat_labels_all[counter]])}')
                        plt.axis('off')
                    else:
                        axs[n2_index, isat_index].imshow(E_clean[counter, data_types_index,:, :], cmap=cmap_types[data_types_index])
                        axs[n2_index, isat_index].set_title(f'power={power_values[power_labels_all[counter]]}, n2={n2_values[n2_labels_all[counter]]}, Isat={"{:e}".format(isat_values[isat_labels_all[counter]])}')
                        plt.axis('off')
                    counter += 1
            plt.tight_layout()
            plt.savefig(f'{path}/{data_types[data_types_index]}_{str(power_values[power_index])[:4]}p_{number_of_n2}n2_{number_of_isat}Isat.png')
            plt.close()