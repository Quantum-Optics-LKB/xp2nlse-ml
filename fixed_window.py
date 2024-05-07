import NLSE.NLSE.nlse as nlse
import numpy as np
from engine.nlse_generator import normalize_data
from engine.waist_fitting import pinhole
from scipy.constants import c, epsilon_0
from skimage.restoration import unwrap_phase
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 50

N = 2048
pixel_size = 5.5e-6
puiss = .4
Isat = 1e6
L = 20e-2
alpha = 18.52504337
pinsize = 2
dz = 1e-5
n2 = -3e-9

path = "/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/input_beam.npy"

print("---- LOAD INPUT IMAGE ----")
image = np.load(path).astype(np.float32)

min_image = np.min(image)
max_image = np.max(image)
image -= min_image
max_image -= min_image
image /= max_image
np.sqrt(image, out=image)

for window in [15e-3, 16e-3, 17e-3, 18e-3, 19e-3, 30e-3]:
    # window = 16e-3
    zoom_factor = pixel_size/(window/N)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 50))

    ax1.imshow(image_init, cmap="viridis", norm=LogNorm())
    ax1.set_title("Before zoom")

    if zoom_factor != 1:
        image = zoom(image_init, zoom_factor, order=5)
    zoomed_shape = image.shape[0]

    if zoomed_shape % 2 == 1:
        image = np.pad(image,(((N - zoomed_shape)//2, (N - zoomed_shape)//2+1),((N - zoomed_shape)//2, (N - zoomed_shape)//2+1)), "constant", constant_values=0)
    else:
        image = np.pad(image,(N - zoomed_shape)//2, "constant", constant_values=0)
    ax2.imshow(image, cmap="viridis", norm=LogNorm())
    ax2.set_title("After zoom")

    plt.tight_layout()
    plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/zoomed.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 50))

    ax1.imshow(image, cmap="viridis", norm=LogNorm()) 
    ax1.set_title("Before pinhole")
    print("---- PINHOLE ----")
    image = pinhole(image, window, N, N, False,pinsize)

    ax2.imshow(image, cmap="viridis", norm=LogNorm()) 
    ax2.set_title("After pinhole")

    plt.tight_layout()
    plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/pinhole_{pinsize}.png")


    image = image + 1j * 0

    simu = nlse.NLSE(alpha, puiss, window, n2, None, L, NX=N , NY=N, Isat=Isat)
    simu.delta_z = dz
    A = simu.out_field(image, L, verbose=True, plot=False, precision="single")

    crop = (N - zoomed_shape)//2
    density = normalize_data(np.abs(A)**2 * c * epsilon_0 / 2)#[crop:N-crop, crop:N-crop]
    #density = zoom(density, 1/zoom_factor, order=5)
    phase = normalize_data(np.angle(A))#[crop:N-crop, crop:N-crop]
    #phase = zoom(phase, 1/zoom_factor, order=5)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 50))

    # Plotting on the first subplot
    ax1.imshow(density, cmap="viridis", norm=LogNorm())
    ax1.set_title("Log Density")

    ax2.imshow(density, cmap="viridis")
    ax2.set_title("Density")

    plt.tight_layout()

    # Show the plot
    plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/findmax_{n2}_w{window}.png")