import NLSE.nlse as nlse
import numpy as np
from scipy.constants import c, epsilon_0
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from engine.generate_augment import add_model_noise, normalize_data
from scipy.ndimage import zoom
import cupy as cp


plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 20


N = 2048
waist = 2.3e-3
L = 20e-2
alpha = 22
Isat = 76842.93830580918
n2 = -1.2878944035735968e-09
dz = 1e-4
nl_length = 0
puiss = 1.05
window = 10*waist

min_out_camera = 3008
pixel_out_camera = 3.76e-6
device = 0

with cp.cuda.Device(device):

    simu = nlse.NLSE(alpha, puiss, window, n2, None, L, NX=N , NY=N, Isat=Isat,nl_length=nl_length)
    simu.delta_z = dz

    beam = np.ones((simu.NX, simu.NY), dtype=np.complex64)*np.exp(-(simu.XX**2 + simu.YY**2) / waist**2)
    poisson_noise_lam, normal_noise_sigma = 0.1 , 0.01
    beam = add_model_noise(beam, poisson_noise_lam, normal_noise_sigma)
    A = simu.out_field(beam, L, verbose=True, plot=False, precision="single")

    density = np.abs(A)**2 * c * epsilon_0 / 2
    density =normalize_data(density)

    fourier = np.abs(np.fft.fftshift(np.fft.fft2(A)))
    fourier = normalize_data(fourier)

    phase = np.angle(A)
    uphase = unwrap_phase(phase)
    uphase = normalize_data(uphase)
    phase = normalize_data(phase)

    output_shape = np.array(phase.shape)
    window_out = min_out_camera*pixel_out_camera

    crop = int(0.5*(window - window_out)*N/window)
    phase = phase[crop:-crop, crop:-crop]
    uphase = uphase[crop:-crop, crop:-crop]
    density = density[crop:-crop, crop:-crop]
    fourier = fourier[crop:-crop, crop:-crop]


    output_shape = np.asarray(uphase.shape)

    label_x = np.around(np.asarray([-window_out/2, 0.0 ,window_out/2])/1e-3,2)
    label_y = np.around(np.asarray([-window_out/2, 0.0 ,window_out/2])/1e-3,2)

    exp_path="/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp/experiment.npy"
    field = np.load(exp_path)

    density_experiment = np.abs(field)
    density_experiment = normalize_data(zoom(density_experiment, 
                    (output_shape[-2]/field.shape[-2], output_shape[-1]/field.shape[-1]))).astype(np.float16)
    
    phase_experiment = np.angle(field)
    uphase_experiment = normalize_data(zoom(unwrap_phase(phase_experiment), 
                    (output_shape[-2]/field.shape[-2], output_shape[-1]/field.shape[-1]))).astype(np.float16)
    phase_experiment = normalize_data(zoom(phase_experiment, 
                    (output_shape[-2]/field.shape[-2], output_shape[-1]/field.shape[-1]))).astype(np.float16)
    fourier_experiment = normalize_data(zoom(np.abs(np.fft.fftshift(np.fft.fft2(field))), 
                    (output_shape[-2]/field.shape[-2], output_shape[-1]/field.shape[-1]))).astype(np.float16)

fig, axs = plt.subplots(4, 2, figsize=(20, 40))

n2_str = r"$n_2$"
n2_u = r"$m^2$/$W$"
isat_str = r"$I_{sat}$"
isat_u = r"$W$/$m^2$"
puiss_str = r"$p$"
puiss_u = r"$W$"
nl_str = r"$nl$"
nl_u = r"$m$"

fig.suptitle(f"{n2_str} = {n2:.2e}{n2_u}, {isat_str} = {Isat:.2e}{isat_u}, {puiss_str} = {puiss:.2e}{puiss_u}, {nl_str} = {nl_length:.2e}{nl_u}")

im1 = axs[0, 0].imshow(density, cmap="viridis")
im2 = axs[1, 0].imshow(phase, cmap="twilight_shifted")
im3 = axs[2, 0].imshow(uphase, cmap="viridis")
im4 = axs[3, 0].imshow(fourier, cmap="viridis")
im5 = axs[0, 1].imshow(density_experiment, cmap="viridis")
im6 = axs[1, 1].imshow(phase_experiment, cmap="twilight_shifted")
im7 = axs[2, 1].imshow(uphase_experiment, cmap="viridis")
im8 = axs[3, 1].imshow(fourier_experiment, cmap="viridis")


divider1 = make_axes_locatable(axs[0, 0])
divider2 = make_axes_locatable(axs[1, 0])
divider3 = make_axes_locatable(axs[2, 0])
divider4 = make_axes_locatable(axs[3, 0])
divider5 = make_axes_locatable(axs[0, 1])
divider6 = make_axes_locatable(axs[1, 1])
divider7 = make_axes_locatable(axs[2, 1])
divider8 = make_axes_locatable(axs[3, 1])

cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cax3 = divider3.append_axes("right", size="5%", pad=0.05)
cax4 = divider4.append_axes("right", size="5%", pad=0.05)
cax5 = divider5.append_axes("right", size="5%", pad=0.05)
cax6 = divider6.append_axes("right", size="5%", pad=0.05)
cax7 = divider5.append_axes("right", size="5%", pad=0.05)
cax8 = divider6.append_axes("right", size="5%", pad=0.05)


cbar1 = plt.colorbar(im1, cax=cax1)
cbar2 = plt.colorbar(im2, cax=cax2)
cbar3 = plt.colorbar(im3, cax=cax3)
cbar4 = plt.colorbar(im4, cax=cax4)
cbar5 = plt.colorbar(im5, cax=cax3)
cbar6 = plt.colorbar(im6, cax=cax4)
cbar7 = plt.colorbar(im5, cax=cax3)
cbar8 = plt.colorbar(im6, cax=cax4)


axs[0, 0].set_title("Density")
axs[0, 0].set_xticks([0, output_shape[1]//2, output_shape[1]])
axs[0, 0].set_xticklabels(label_x)
axs[0, 0].set_xlabel(r"x (mm$^{-1}$)")
axs[0, 0].set_yticks([0, output_shape[0]//2, output_shape[0]])
axs[0, 0].set_yticklabels(label_y)
axs[0, 0].set_ylabel(r"y (mm$^{-1}$)")

axs[1, 0].set_title("Phase")
axs[1, 0].set_xticks([0, output_shape[1]//2, output_shape[1]])
axs[1, 0].set_xticklabels(label_x)
axs[1, 0].set_xlabel(r"x (mm$^{-1}$)")
axs[1, 0].set_yticks([0, output_shape[0]//2, output_shape[0]])
axs[1, 0].set_yticklabels(label_y)
axs[1, 0].set_ylabel(r"y (mm$^{-1}$)")

axs[2, 0].set_title("Phase Unwrap")
axs[2, 0].set_xticks([0, output_shape[1]//2, output_shape[1]])
axs[2, 0].set_xticklabels(label_x)
axs[2, 0].set_xlabel(r"x (mm$^{-1}$)")
axs[2, 0].set_yticks([0, output_shape[0]//2, output_shape[0]])
axs[2, 0].set_yticklabels(label_y)
axs[2, 0].set_ylabel(r"y (mm$^{-1}$)")

axs[3, 0].set_title("Fourier")
# axs[3, 0].set_xticks([0, output_shape[1]//2, output_shape[1]])
# axs[3, 0].set_xticklabels(label_x)
axs[3, 0].set_xlabel(r"kx (mm$^{-1}$)")
# axs[3, 0].set_yticks([0, output_shape[0]//2, output_shape[0]])
# axs[3, 0].set_yticklabels(label_y)
axs[3, 0].set_ylabel(r"ky (mm$^{-1}$)")

axs[0, 1].set_title("Experimental Density")
axs[0, 1].set_xticks([0, output_shape[1]//2, output_shape[1]])
axs[0, 1].set_xticklabels(label_x)
axs[0, 1].set_xlabel(r"x (mm$^{-1}$)")
axs[0, 1].set_yticks([0, output_shape[0]//2, output_shape[0]])
axs[0, 1].set_yticklabels(label_y)
axs[0, 1].set_ylabel(r"y (mm$^{-1}$)")

axs[1, 1].set_title("Experimental Phase")
axs[1, 1].set_xticks([0, output_shape[1]//2, output_shape[1]])
axs[1, 1].set_xticklabels(label_x)
axs[1, 1].set_xlabel(r"x (mm$^{-1}$)")
axs[1, 1].set_yticks([0, output_shape[0]//2, output_shape[0]])
axs[1, 1].set_yticklabels(label_y)
axs[1, 1].set_ylabel(r"y (mm$^{-1}$)")

axs[2, 1].set_title("Experimental Phase Unwrap")
axs[2, 1].set_xticks([0, output_shape[1]//2, output_shape[1]])
axs[2, 1].set_xticklabels(label_x)
axs[2, 1].set_xlabel(r"x (mm$^{-1}$)")
axs[2, 1].set_yticks([0, output_shape[0]//2, output_shape[0]])
axs[2, 1].set_yticklabels(label_y)
axs[2, 1].set_ylabel(r"y (mm$^{-1}$)")

axs[3, 1].set_title("Experimental Fourier")
# axs[3, 1].set_xticks([0, output_shape[1]//2, output_shape[1]])
# axs[3, 1].set_xticklabels(label_x)
axs[3, 1].set_xlabel(r"kx (mm$^{-1}$)")
# axs[3, 1].set_yticks([0, output_shape[0]//2, output_shape[0]])
# axs[3, 1].set_yticklabels(label_y)
axs[3, 1].set_ylabel(r"ky (mm$^{-1}$)")



axs[0, 0].tick_params(axis='both', which='major', pad=15)
axs[1, 0].tick_params(axis='both', which='major', pad=15)
axs[2, 0].tick_params(axis='both', which='major', pad=15)
axs[3, 0].tick_params(axis='both', which='major', pad=15)
axs[0, 1].tick_params(axis='both', which='major', pad=15)
axs[1, 1].tick_params(axis='both', which='major', pad=15)
axs[2, 1].tick_params(axis='both', which='major', pad=15)
axs[3, 1].tick_params(axis='both', which='major', pad=15)

plt.tight_layout()
plt.savefig(f"/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/output_nl{nl_length:.2e}_p{puiss:.2e}_n2{n2:.1e}_isat{Isat:.1e}.png")