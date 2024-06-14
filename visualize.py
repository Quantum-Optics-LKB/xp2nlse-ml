import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 20


def plot_and_save_images(data, path, number_of_n2, number_of_isat):

    n2 = -np.logspace(-10, -8, number_of_n2) #m/W^2
    isat = np.logspace(4, 6, number_of_isat) #W/m^2
    power= 1.05

    # Separate density and phase channels
    data = data.reshape((number_of_n2, number_of_isat, 3, data.shape[-2], data.shape[-1]))
    Nn2, Nisat, Nchannels, resolution, _ = data.shape

    density_channels = data[:, :, 0, :, :]
    phase_channels = data[:, :, 1, :, :]
    uphase_channels = data[:, :, 2, :, :]
    
    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    
    fig_density, axes_density = plt.subplots(Nn2, Nisat, figsize=(50,50))
    fig_density.suptitle(f'Density Channels - {puiss_str} = {power:.2e} {puiss_u}')

    for n in range(Nn2):
        for i in range(Nisat):
            ax = axes_density if Nn2 == 1 and Nisat == 1 else (axes_density[n, i] if Nn2 > 1 and Nisat > 1 else (axes_density[n] if Nn2 > 1 else axes_density[i]))
            ax.imshow(density_channels[n, i, :, :], cmap='viridis')
            ax.set_title(f'{n2_str} = {n2[n]:.2e} {n2_u},\n{isat_str} = {isat[i]:.2e} {isat_u}')
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{path}density_power_{power}.png')
    plt.close(fig_density) 

    # Plot phase channels for the current power value
    fig_phase, axes_phase = plt.subplots(Nn2, Nisat, figsize=(50, 50))
    fig_phase.suptitle(f'Phase Channels - {puiss_str} = {power:.2e} {puiss_u}')

    for n in range(Nn2):
        for i in range(Nisat):
            ax = axes_phase if Nn2 == 1 and Nisat == 1 else (axes_phase[n, i] if Nn2 > 1 and Nisat > 1 else (axes_phase[n] if Nn2 > 1 else axes_phase[i]))
            ax.imshow(phase_channels[n, i, :, :], cmap='twilight_shifted')
            ax.set_title(f'{n2_str} = {n2[n]:.2e} {n2_u},\n{isat_str} = {isat[i]:.2e} {isat_u}')
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{path}phase_power_{power}.png')
    plt.close(fig_phase)

    #Plot phase channels for the current power value
    fig_phase, axes_phase = plt.subplots(Nn2, Nisat, figsize=(50,50))
    fig_phase.suptitle(f'Unwrap Phase Channels - {puiss_str} = {power:.2e} {puiss_u}')

    for n in range(Nn2):
        for i in range(Nisat):
            ax = axes_phase if Nn2 == 1 and Nisat == 1 else (axes_phase[n, i] if Nn2 > 1 and Nisat > 1 else (axes_phase[n] if Nn2 > 1 else axes_phase[i]))
            ax.imshow(uphase_channels[n, i, :, :], cmap='viridis')
            ax.set_title(f'{n2_str} = {n2[n]:.2e} {n2_u},\n{isat_str} = {isat[i]:.2e} {isat_u}')
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{path}unwrap_phase_power_{power}.png')
    plt.close(fig_phase)
power = 1.05
number_of_n2, number_of_isat = 5, 5
path = "/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/"
data = np.load(f"{path}Es_w256_n2{number_of_n2}_isat{number_of_isat}_power{power:.2f}.npy")

plot_and_save_images(data,path,number_of_n2, number_of_isat)
