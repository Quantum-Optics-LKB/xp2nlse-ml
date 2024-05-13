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
n2 = -1e-9


path = "/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/04191653_time_flat_real/input_beam.npy"

print("---- LOAD INPUT IMAGE ----")
image = np.load(path).astype(np.float32)

min_image = np.min(image)
max_image = np.max(image)
image -= min_image
max_image -= min_image
image /= max_image
np.sqrt(image, out=image)

window = 35e-3

padded_shape = int(round((window/pixel_size)/2)*2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 50))

ax1.imshow(image, cmap="viridis", norm=LogNorm())
ax1.set_title("Before pad")

image = np.pad(image, (padded_shape - N)//2, 'constant', constant_values=0)

ax2.imshow(image, cmap="viridis", norm=LogNorm())
ax2.set_title("After pad")

plt.tight_layout()
plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/padded.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 50))

ax1.imshow(image, cmap="viridis", norm=LogNorm()) 
ax1.set_title("Before pinhole")
print("---- PINHOLE ----")
image = pinhole(image, window, padded_shape, padded_shape, False,pinsize)

ax2.imshow(image, cmap="viridis", norm=LogNorm()) 
ax2.set_title("After pinhole")

plt.tight_layout()
plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/pinhole_{pinsize}.png")

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 50))

# ax1.imshow(image, cmap="viridis", norm=LogNorm()) 
# ax1.set_title("Before zoom")
# zoom_factor = N/padded_shape
# image = zoom(image, zoom_factor)
# ax2.imshow(image, cmap="viridis") 
# ax2.set_title("After zoom")

# plt.tight_layout()
# plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/zoom.png")


image = image + 1j * 0

simu = nlse.NLSE(alpha, puiss, window, n2, None, L, NX=padded_shape , NY=padded_shape, Isat=Isat)
simu.delta_z = dz
A = simu.out_field(image, L, verbose=True, plot=False, precision="single")

crop = (padded_shape - N)//2
density_init = normalize_data(np.abs(A)**2 * c * epsilon_0 / 2)
density = density_init[crop:padded_shape-crop, crop:padded_shape-crop]
# density = zoom(density_init, 1/zoom_factor, order=5)
phase_init = normalize_data(np.angle(A))
phase = phase_init[crop:padded_shape-crop, crop:padded_shape-crop]
# phase = zoom(phase_init, 1/zoom_factor, order=5)[crop:padded_shape-crop, crop:padded_shape-crop]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 50))

# Plotting on the first subplot
ax1.imshow(density, cmap="viridis", norm=LogNorm())
ax1.set_title("Log Density")

ax2.imshow(density, cmap="viridis")
ax2.set_title("Density")

plt.tight_layout()

# Show the plot
plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/findmax_{n2}_w{window}.png")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 50))

# Plotting on the first subplot
ax1.imshow(density_init, cmap="viridis", norm=LogNorm())
ax1.set_title("Log Density")

ax2.imshow(density_init, cmap="viridis")
ax2.set_title("Density")

plt.tight_layout()

# Show the plot
plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/findmax_{n2}_w{window}_init.png")