import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 7

# Create a figure with 4 subplots in a 2x3 grid
fig, axs = plt.subplots(3, 2, figsize=(9, 15))

# Generate data
smallest_out_res = 3008
out_pixel_size = 3.76e-6
window_out = out_pixel_size * smallest_out_res
x = np.linspace(-window_out/2, window_out/2, smallest_out_res)
y = np.linspace(-window_out/2, window_out/2, smallest_out_res)
x, y = np.meshgrid(x, y)

# Load data
z = np.abs(np.load("/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/experiment.npy"))

# Plot the original data
im1 = axs[0, 0].imshow(z, extent=[-window_out/2, window_out/2, -window_out/2, window_out/2], cmap='viridis', origin='lower')
axs[0, 0].set_title('Original Data')
axs[0, 0].set_xlabel('X Label')
axs[0, 0].set_ylabel('Y Label')
fig.colorbar(im1, ax=axs[0, 0])

# Define a Gaussian function
def gaussian(xy, amp, xo, yo, sigma, k):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    g = amp * np.exp(-2*((x - xo)**2 + (y - yo)**2) / sigma**2)**k
    return g.ravel()

# Create initial guess for the Gaussian parameters
initial_guess = (np.max(z), 0, 0, 3e-3, 1.5)

# Perform Gaussian fit
xdata = np.vstack((x.ravel(), y.ravel()))
popt, pcov = curve_fit(gaussian, xdata, z.ravel(), p0=initial_guess)
print(popt)
fit = gaussian(xdata, *popt).reshape(smallest_out_res, smallest_out_res)

# Plot the fitted Gaussian
im2 = axs[0, 1].imshow(fit, extent=[-window_out/2, window_out/2, -window_out/2, window_out/2], cmap='viridis', origin='lower')
axs[0, 1].set_title('Fitted Gaussian')
axs[0, 1].set_xlabel('X Label')
axs[0, 1].set_ylabel('Y Label')
fig.colorbar(im2, ax=axs[0, 1])

# Calculate and plot the difference
# z -= np.min(z)
# z /= np.max(z)

# fit -= np.min(fit)
# fit /= np.max(fit)

noise = z - fit
im3 = axs[1, 0].imshow(np.abs(noise), extent=[-window_out/2, window_out/2, -window_out/2, window_out/2], 
                       cmap='viridis', origin='lower', 
                       norm=LogNorm())
axs[1, 0].set_title('Difference (Original - Fitted)')
axs[1, 0].set_xlabel('X Label')
axs[1, 0].set_ylabel('Y Label')
fig.colorbar(im3, ax=axs[1, 0])

# Plot the sum of the difference and the original data
sum_diff_original = noise*fit
im4 = axs[1, 1].imshow(sum_diff_original, extent=[-window_out/2, window_out/2, -window_out/2, window_out/2], cmap='viridis', origin='lower')
axs[1, 1].set_title('Sum (Difference + Original)')
axs[1, 1].set_xlabel('X Label')
axs[1, 1].set_ylabel('Y Label')
fig.colorbar(im4, ax=axs[1, 1])

# Cut view of the experimental data
cut_idx = smallest_out_res // 2
axs[2, 0].plot(x[cut_idx], z[cut_idx])
axs[2, 0].set_title('Cut View of Experimental Data')
axs[2, 0].set_xlabel('X Label')
axs[2, 0].set_ylabel('Intensity')

# Cut view of the fitted Gaussian
axs[2, 1].plot(x[cut_idx], z[cut_idx], label="experimental")
axs[2, 1].plot(x[cut_idx], fit[cut_idx], label="fit")
axs[2, 1].set_title('Cut View of Fitted Gaussian')
axs[2, 1].set_xlabel('X Label')
axs[2, 1].set_ylabel('Intensity')
axs[2, 1].legend()

plt.tight_layout()
plt.savefig("test.png")
