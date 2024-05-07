import NLSE.NLSE.nlse as nlse
import numpy as np
from engine.nlse_generator import normalize_data
from engine.waist_fitting import pinhole
from scipy.constants import c, epsilon_0
from skimage.restoration import unwrap_phase
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 50

pixel_size = 5.5e-6
puiss = .4
Isat = 1e6
L = 20e-2
alpha = 18.52504337
pinsize = 2
dz = 1e-5



path = "/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/input_beam.npy"


for n2, pad_width in zip([-2e-9], [2048]):

    print("---- LOAD INPUT IMAGE ----")
    image = np.load(path).astype(np.float32)
    N = image.shape[0]

    image = (image - np.min(image))/(np.max(image) - np.min(image))
    image = np.sqrt(image)
    
    zoom_factor = 1
    # pad_width = int((N - N*zoom_factor)/2)
    window = N * pixel_size + 2*pad_width*pixel_size*(1/zoom_factor)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 50))

    im1 = ax1.imshow(image, cmap="viridis", norm=LogNorm()) 
    ax1.set_title("Before zoom")

    if zoom_factor != 1:
        image = zoom(image, zoom_factor, order=5)
    N = int(N * zoom_factor)

    im2 = ax2.imshow(image, cmap="viridis", norm=LogNorm()) 
    ax2.set_title("After zoom")

    plt.tight_layout()
    plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/zoom_{zoom_factor}.png")
    
    image = np.pad(image, pad_width,"edge")# "constant", constant_values=0)
    N = pad_width*2 + N
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 50))

    ax1.imshow(image, cmap="viridis", norm=LogNorm()) 
    ax1.set_title("Before pinhole")

    print("---- PINHOLE ----")
    image = pinhole(image, window, N, N, False,pinsize)
    print(image.shape)

    ax2.imshow(image, cmap="viridis", norm=LogNorm()) 
    ax2.set_title("After pinhole")

    plt.tight_layout()

    plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/pinhole_{pinsize}.png")

    image = image + 1j * 0

    simu = nlse.NLSE(alpha, puiss, window, n2, None, L, NX=N , NY=N, Isat=Isat)
    simu.delta_z = dz

    A = simu.out_field(image, L, verbose=True, plot=False, precision="single")[pad_width:N-pad_width,pad_width:N-pad_width]

    density = normalize_data(np.abs(A)**2 * c * epsilon_0 / 2)
    phase = normalize_data(np.angle(A))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 50))

    # Plotting on the first subplot
    ax1.imshow(density, cmap="viridis", norm=LogNorm())
    ax1.set_title("Log Density")

    ax2.imshow(density, cmap="viridis")
    ax2.set_title("Density")

    plt.tight_layout()

    # Show the plot
    plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/findmax_{n2}_w{window}.png")