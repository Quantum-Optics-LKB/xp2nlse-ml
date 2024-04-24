from matplotlib import pyplot as plt
import NLSE
import numpy as np
import pyfftw
from scipy.constants import c, epsilon_0
from PIL import Image
from scipy.ndimage import zoom
from engine.waist_fitting import pinhole
from skimage.restoration import unwrap_phase

import cupy as cp



def from_input_image(
        path: str,
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
    print("---- LOAD INPUT IMAGE ----")
    # input_tiff = Image.open(path)
    
    # image = np.array(input_tiff, dtype=np.float32)

    image = np.load(path).astype(np.float32)
    image = (image- np.min(image))/(np.max(image) -np.min(image))
    image = np.sqrt(image)
    resolution_image = image.shape[0]
    
    if resolution_in != image.shape[0]:
        image = zoom(image, (resolution_in/image.shape[0],resolution_in/image.shape[1]))

    print("---- PINHOLE ----")
    window =  resolution_image * 5.5e-6 
    image = pinhole(image, window, image.shape[0], image.shape[0], False, 1)

    image = image + 1j * 0

    print("---- PREPARE FOR NLSE ----")
    input_field = image.astype(np.complex64)

    input_field_tiled_n2_power_isat = np.tile(input_field[np.newaxis,np.newaxis, np.newaxis, :,:], (1,1,1, 1,1))
    return input_field_tiled_n2_power_isat, window

# for integration testing
def main():

    N = 1024
    n2_value = -1e-9#-7.3e-10
    window = N*5.5e-6
    power_value = .4 #0.5
    Isat_value = 1e6  #5.9e5
    L = 20e-2
    alpha_value = 20 #23 # 
    delta_z = 1e-5

    n2 =  cp.zeros((1 ,1 ,1 ,1 ,1))
    n2[:, 0, 0, 0, 0] = cp.array([n2_value])
    # n2 = n2_value

    power =  cp.zeros((1, 1 ,1 ,1 ,1))
    power[0, :, 0, 0, 0] = cp.array([power_value])
    # power = power_value

    Isat =  cp.zeros((1, 1, 1 ,1 ,1 ))
    Isat[0, 0, :, 0, 0] = cp.array([Isat_value])

    # Isat = Isat_value
    alpha = alpha_value

    
    

    E_0, window = from_input_image("/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/input_beam.npy", N)
    simu = NLSE.nlse.NLSE(alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat)
    simu.delta_z = delta_z
    A = simu.out_field(cp.array(E_0), L, verbose=True, plot=False, precision="double").get()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display first image
    axes[0].imshow(np.abs(E_0[0, 0, 0, :, :])**2 * c * epsilon_0 / 2)
    axes[0].axis('off')  # Turn off axis
    axes[0].set_title('Density')

    # Display second image
    axes[1].imshow(unwrap_phase(np.angle(E_0[0, 0, 0, :, :]), rng=0))
    axes[1].axis('off')  # Turn off axis
    axes[1].set_title('Phase')

    # Display the plot
    plt.tight_layout()
    plt.savefig("input.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display first image
    axes[0].imshow(np.abs(A[0, 0, 0, :, :])**2 * c * epsilon_0 / 2)
    axes[0].axis('off')  # Turn off axis
    axes[0].set_title('Density')

    # Display second image
    axes[1].imshow(unwrap_phase(np.angle(A[0, 0, 0, :, :]), rng=0))
    axes[1].axis('off')  # Turn off axis
    axes[1].set_title('Phase')

    # Display the plot
    plt.tight_layout()
    plt.savefig("output.png")
    plt.close()


if __name__ == "__main__":
    main()
