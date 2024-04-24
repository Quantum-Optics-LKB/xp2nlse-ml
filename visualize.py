import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from engine.waist_fitting import pinhole

# Example dimensions
Nn2 = 2
NIsat = 2
Npower = 20

n2_values = np.linspace(-1e-11, -1e-9, Nn2)
n2_labels = np.arange(0, Nn2)

isat_values = np.linspace(1e4, 1e6, NIsat)
isat_labels = np.arange(0, NIsat)

N2_values_single, ISAT_values_single = np.meshgrid(n2_values, isat_values,) 

n2_values_all_single = N2_values_single.flatten()
isat_values_all_single = ISAT_values_single.flatten()

def crop_largest_square_from_array(img_array):
    """
    Crops the largest square from a numpy array that excludes all rows and columns entirely filled with zeros.

    :param img_array: A numpy array representing the image, with values clipped between 0 and 1.
    """
    # Validate that the input is a square matrix
    if img_array.shape[0] != img_array.shape[1]:
        raise ValueError("Input array must be square (equal width and height).")

    # Check where the rows and columns have any non-zero values
    non_zero_columns = np.where(img_array.max(axis=0) > 0)[0]
    non_zero_rows = np.where(img_array.max(axis=1) > 0)[0]

    # If no non-zero columns or rows, return None or a minimal array
    if len(non_zero_columns) == 0 or len(non_zero_rows) == 0:
        return np.array([[0]])

    # Determine the bounds of the non-zero area
    left = non_zero_columns.min()
    right = non_zero_columns.max()
    top = non_zero_rows.min()
    bottom = non_zero_rows.max()

    # Determine the maximum size of the square
    square_size = min(right - left, bottom - top) + 1

    # Calculate new bounds centered within the non-zero area to form a square
    new_left = left
    new_right = left + square_size
    new_top = top
    new_bottom = top + square_size

    # Adjust bounds if the calculated dimensions go beyond the image's limits
    if new_right > img_array.shape[1]:
        new_right = img_array.shape[1]
        new_left = img_array.shape[1] - square_size
    if new_bottom > img_array.shape[0]:
        new_bottom = img_array.shape[0]
        new_top = img_array.shape[0] - square_size

    # Crop the array
    cropped_array = img_array[new_top:new_bottom, new_left:new_right]

    img = zoom(img_array.astype(np.float64), (256/cropped_array.shape[0], 256/cropped_array.shape[1]), order=5)
    return cropped_array


# Simulated data: Replace this with your actual numpy array
data = np.load(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/Es_w256_n2{Nn2}_isat{NIsat}_power{Npower}.npy")
even_indices = np.arange(0, Npower*2, 2)
data_even = data[:, even_indices, :, :]

odd_indices = np.arange(1, Npower*2, 2)
data_odd = data[:, odd_indices, :, :]
# Plotting

for power in range(Npower):
    fig, axes = plt.subplots(Nn2, NIsat, figsize=(Nn2*5, NIsat*5))

    index_n2_isat = 0

    for i in range(Nn2):
        for j in range(NIsat):
            ax = axes[i, j] if Nn2 * NIsat > 1 else axes
            density = np.copy(data_even[index_n2_isat, power, :,:])
            im = ax.imshow(density, cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.axis('off')
            ax.set_title(f'n2 = {n2_values_all_single[index_n2_isat]:2e}\n Isat =  {isat_values_all_single[index_n2_isat]:2e}')
            index_n2_isat += 1

    plt.tight_layout()
    plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/Density_w256_n2{Nn2}_isat{NIsat}_power{Npower}_{power}.png")
    plt.close()

    fig, axes = plt.subplots(Nn2, NIsat, figsize=(Nn2*5, NIsat*5))

    index_n2_isat = 0

    for i in range(Nn2):
        for j in range(NIsat):
            ax = axes[i, j] if Nn2 * NIsat > 1 else axes
            density = np.copy(data_even[index_n2_isat, power, :,:])
            phase = np.copy(data_odd[index_n2_isat, power, :,:])
            im = ax.imshow(phase, cmap='viridis')
            plt.colorbar(im, ax=ax )
            ax.axis('off')
            ax.set_title(f'n2 = {n2_values_all_single[index_n2_isat]:2e}\n Isat =  {isat_values_all_single[index_n2_isat]:2e}')
            index_n2_isat += 1

    plt.tight_layout()
    plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/Phase_w256_n2{Nn2}_isat{NIsat}_power{Npower}_{power}.png")
    plt.close()
