import numpy as np
import matplotlib.pyplot as plt
from contrast import disk
# from perlin_numpy import generate_fractal_noise_2d
from scipy import ndimage
from skimage import data
import contrast as contrast

np.random.seed(1)

N = 512
x = np.linspace(-N // 2, N // 2, N)
y = np.linspace(-N // 2, N // 2, N)
X, Y = np.meshgrid(x, y)
E_r = np.exp(-(X**2 + Y**2) / (4 * N**2))
noise_amp = 1e-2
noise_size = 256
# noise_r = generate_fractal_noise_2d((N, N), (noise_size, noise_size))
# noise_s = generate_fractal_noise_2d((N, N), (noise_size, noise_size))
noise_r = np.random.normal(0, 1, (N, N))
noise_s = np.random.normal(0, 1, (N, N))
kx = np.fft.rfftfreq(N)[N // 4] * 2 * np.pi
ky = kx
E_r += noise_amp * noise_r
E_r = E_r * np.exp(1j * (kx * X + ky * Y))
E_s = np.exp(-(X**2 + Y**2) / N**2)
E_s *= data.astronaut()[:, :, 0] / 255
E_s += noise_amp * noise_s
interf = np.abs(E_r + E_s) ** 2
interf_fft = np.fft.rfft2(interf)
im_fft = np.abs(interf_fft)
# plt.imshow(
#     im_fft,
#     cmap="nipy_spectral",
#     vmax=2.5e-3 * np.nanmax(im_fft),
# )
# plt.show()
if ky == 2 * kx:
    interf_ifft = interf_fft[N // 4 : -N // 4, :-1]
    interf_ifft *= disk(N // 2, N // 2, (N // 4, N // 4), N // 4)
    field = contrast.im_osc_fast_t(interf, center=(N // 2, N // 4))

elif ky == kx:
    interf_ifft = interf_fft[: N // 2, :-1]
    interf_ifft *= disk(N // 2, N // 2, (N // 4, N // 4), N // 4)
    field = contrast.im_osc_fast_t(interf, center=(N // 4, N // 4))


# plt.imshow(
#     np.abs(interf_ifft),
#     cmap="nipy_spectral",
#     vmax=5e-3 * np.nanmax(np.abs(interf_ifft)),
# )
# plt.show()
interf_ifft = np.fft.fftshift(interf_ifft)
# field = np.fft.ifft2(interf_ifft)
I_s = np.abs(E_s) ** 2
I_r = np.abs(E_r) ** 2
I_s = ndimage.zoom(I_s, 0.5, order=0)
I_r = ndimage.zoom(I_r, 0.5, order=0)
recov = np.abs(field) ** 2 / I_r
err = recov - I_s
print(np.mean(np.abs(err) ** 2))
fig, ax = plt.subplots(1, 3)
ax[0].set_title("Recovered")
ax[0].imshow(recov)
ax[1].set_title("Initial down sampled")
ax[1].imshow(I_s)
ax[2].set_title("Difference")
ax[2].imshow(err, cmap="seismic")
plt.show()
