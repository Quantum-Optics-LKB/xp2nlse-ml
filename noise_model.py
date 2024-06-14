import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import welch

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 7

# Load the experimental data
data_path = "/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/experiment.npy"
data = np.abs(np.load(data_path))

# Analyze the noise in the experimental data
signal_estimate = gaussian_filter(data, sigma=10)
noise = data - signal_estimate

# Calculate the statistical properties of the noise
noise_mean = np.mean(noise)
noise_std = np.std(noise)

# Calculate the Power Spectral Density (PSD) of the noise
freqs, psd = welch(noise.ravel())

# Create a synthetic noise model
def create_noise_model(shape, noise_mean, noise_std, psd):
    # Gaussian noise component
    gaussian_noise = np.random.normal(loc=noise_mean.real, scale=noise_std, size=shape) + \
                     1j * np.random.normal(loc=noise_mean.imag, scale=noise_std, size=shape)

    # Frequency-dependent noise component
    noise_fft = np.fft.fft2(gaussian_noise)

    # Create a 2D psd_filter from 1D psd_real and psd_imag
    psd_filter = np.sqrt(psd[:shape[0]//2+1])
    psd_filter_2d = np.outer(psd_filter, np.ones(shape[1]))

    # Create the full 2D filter for all frequencies
    psd_filter_2d_full = np.zeros_like(noise_fft)
    psd_filter_2d_full[:shape[0]//2+1, :] = psd_filter_2d
    psd_filter_2d_full[-(shape[0]//2):, :] = psd_filter_2d[-2:0:-1, :]

    # Apply the PSD as a filter to the noise in the frequency domain
    noise_fft_filtered = noise_fft * psd_filter_2d_full

    # Inverse FFT to get the time-domain noise
    filtered_noise = np.fft.ifft2(noise_fft_filtered)

    return filtered_noise

# Define a Gaussian function
def gaussian(xy, amp, xo, yo, sigma):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    g = amp * np.exp(-((x - xo)**2 + (y - yo)**2) / sigma**2)
    return g.ravel()

# Create initial guess for the Gaussian parameters
smallest_out_res = 3008
out_pixel_size = 3.76e-6
window_out = out_pixel_size * smallest_out_res
x = np.linspace(-window_out/2, window_out/2, smallest_out_res)
y = np.linspace(-window_out/2, window_out/2, smallest_out_res)
x, y = np.meshgrid(x, y)

initial_guess = (np.max(data), 0, 0, 3e-3)

# Perform Gaussian fit
xdata = np.vstack((x.ravel(), y.ravel()))
popt, pcov = curve_fit(gaussian, xdata, data.ravel(), p0=initial_guess)
fit = gaussian(xdata, *popt).reshape(smallest_out_res, smallest_out_res)

# Generate synthetic noise
shape = data.shape
synthetic_noise = create_noise_model(shape, noise_mean, noise_std, psd)

# Add synthetic noise to the Gaussian fit
noisy_gaussian_fit = fit + synthetic_noise

# Plot the results
fig, axs = plt.subplots(2, 2, figsize=(9, 9))

# Plot the initial image
axs[0, 0].imshow(np.abs(data), cmap='viridis')
axs[0, 0].set_title('Initial Image')
axs[0, 0].set_xlabel('X Label')
axs[0, 0].set_ylabel('Y Label')

# Plot the initial Gaussian fit
axs[0, 1].imshow(np.abs(fit), cmap='viridis')
axs[0, 1].set_title('Initial Gaussian Fit')
axs[0, 1].set_xlabel('X Label')
axs[0, 1].set_ylabel('Y Label')

# Plot the noise model
axs[1, 0].imshow(np.abs(synthetic_noise), cmap='viridis')
axs[1, 0].set_title('Noise Model')
axs[1, 0].set_xlabel('X Label')
axs[1, 0].set_ylabel('Y Label')

# Plot the noisy Gaussian fit
axs[1, 1].imshow(np.abs(noisy_gaussian_fit), cmap='viridis')
axs[1, 1].set_title('Noisy Gaussian Fit')
axs[1, 1].set_xlabel('X Label')
axs[1, 1].set_ylabel('Y Label')

plt.tight_layout()
plt.show()
