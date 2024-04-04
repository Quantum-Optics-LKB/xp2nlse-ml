from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
from nlse_generator import create, expend
from PIL import Image
from waist_fitting import waist_computation

def from_input_image(path, number_of_power, number_of_n2, number_of_isat, resolution_in):
    print("--- LOAD INPUT IMAGE ---")
    input_tiff = Image.open(path)
    tiff_to_array = np.array(input_tiff, dtype=np.float32)

    print("--- FIND WAIST ---")
    window = resolution_in * 5.5e-6
    # waist = waist_computation(tiff_to_array, window, resolution_in, resolution_in, False)
    waist = 0.00043868048364296215

    print("--- PREPARE FOR NLSE ---")
    # array_normalized = (tiff_to_array - np.min(tiff_to_array) )/ (np.max(tiff_to_array) - np.min(tiff_to_array))
    input_field = tiff_to_array + 1j * np.zeros_like(input_tiff, dtype=np.float32)
    input_field_tiled_n2_power_isat = np.tile(input_field[np.newaxis,np.newaxis, np.newaxis, :,:], (number_of_n2,number_of_power,number_of_isat, 1,1))

    return input_field_tiled_n2_power_isat, waist

def from_gaussian(NX, NY, window, waist ):
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
resolution_in = 2048
resolution_out = 512

number_of_n2 = 5
number_of_power = 3
number_of_isat = 5

is_from_image = True
visualize = True
expension = False
generate = True
precision = np.complex64
delta_z = 1e-1

n2_values = np.linspace(-1e-11, -1e-10, number_of_n2)
n2_labels = np.arange(0, number_of_n2)

power_values = np.linspace(.02, 0.5001, number_of_power)
power_labels = np.arange(0, number_of_power)

isat_values = np.linspace(1e5, 5e5, number_of_isat)
isat_labels = np.arange(0, number_of_isat)

N2_values, POWER_values, ISAT_values = np.meshgrid(n2_values,power_values, isat_values,) 
N2_labels, POWER_labels, ISAT_labels = np.meshgrid(n2_labels, power_labels, isat_labels)

power_values_all = POWER_values.reshape((number_of_power*number_of_n2*number_of_isat,))
n2_values_all = N2_values.reshape((number_of_power*number_of_n2*number_of_isat,))
isat_values_all = ISAT_values.reshape((number_of_power*number_of_n2*number_of_isat,))

power_labels_all = POWER_labels.reshape((number_of_power*number_of_n2*number_of_isat,))
n2_labels_all = N2_labels.reshape((number_of_power*number_of_n2*number_of_isat,))
isat_labels_all = ISAT_labels.reshape((number_of_power*number_of_n2*number_of_isat,))

factor_window = 10
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
        E_clean = create(input_field, window, n2_values,power_values,isat_values, resolution_in,resolution_out, delta_z,path)
else:
    E_clean = np.load(f"{path}/Es_w{resolution_out}_n2{number_of_n2}_Isat{number_of_isat}_power{number_of_power}_amp_pha_pha_unwrap_all.npy")

if expension:
    noise = 0.01
    power = 0
    E_expend = expend(number_of_n2, number_of_power, number_of_isat, power, E_clean, noise, path)

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