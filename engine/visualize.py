import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 20
from seed_settings import set_seed
set_seed(42)

def plot_and_save_images(data, saving_path, nlse_settings):
    n2, input_power, alpha, isat, waist_input_beam, non_locality_length, delta_z, cell_length = nlse_settings
    number_of_n2 = len(n2)
    number_of_isat = len(isat)

    density_channels = data[:, :, 0, :, :]
    phase_channels = data[:, :, 1, :, :]
    uphase_channels = data[:, :, 2, :, :]
    
    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    
    fig_density, axes_density = plt.subplots(number_of_n2, number_of_isat, figsize=(10*number_of_isat,10*number_of_n2))
    fig_density.suptitle(f'Density Channels - {puiss_str} = {input_power:.2e} {puiss_u}')

    for n in range(number_of_n2):
        for i in range(number_of_isat):
            ax = axes_density if number_of_n2 == 1 and number_of_isat == 1 else (axes_density[n, i] if number_of_n2 > 1 and number_of_n2 > 1 else (axes_density[n] if number_of_n2 > 1 else axes_density[i]))
            ax.imshow(density_channels[n, i, :, :], cmap='viridis')
            ax.set_title(f'{n2_str} = {n2[n]:.2e} {n2_u},\n{isat_str} = {isat[i]:.2e} {isat_u}')
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{saving_path}density_n2{number_of_n2}_isat{number_of_isat}_power{input_power:.2f}.png')
    plt.close(fig_density) 

    # Plot phase channels for the current power value
    fig_phase, axes_phase = plt.subplots(number_of_n2, number_of_isat, figsize=(10*number_of_isat,10*number_of_n2))
    fig_phase.suptitle(f'Phase Channels - {puiss_str} = {input_power:.2e} {puiss_u}')

    for n in range(number_of_n2):
        for i in range(number_of_isat):
            ax = axes_phase if number_of_n2 == 1 and number_of_isat == 1 else (axes_phase[n, i] if number_of_n2 > 1 and number_of_isat > 1 else (axes_phase[n] if number_of_n2 > 1 else axes_phase[i]))
            ax.imshow(phase_channels[n, i, :, :], cmap='twilight_shifted')
            ax.set_title(f'{n2_str} = {n2[n]:.2e} {n2_u},\n{isat_str} = {isat[i]:.2e} {isat_u}')
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{saving_path}phase_n2{number_of_n2}_isat{number_of_isat}_power{input_power:.2f}.png')
    plt.close(fig_phase)

    #Plot phase channels for the current power value
    fig_phase, axes_phase = plt.subplots(number_of_n2, number_of_isat, figsize=(50,50))
    fig_phase.suptitle(f'Unwrap Phase Channels - {puiss_str} = {input_power:.2e} {puiss_u}')

    for n in range(number_of_n2):
        for i in range(number_of_isat):
            ax = axes_phase if number_of_n2 == 1 and number_of_isat == 1 else (axes_phase[n, i] if number_of_n2 > 1 and number_of_isat > 1 else (axes_phase[n] if number_of_n2 > 1 else axes_phase[i]))
            ax.imshow(uphase_channels[n, i, :, :], cmap='viridis')
            ax.set_title(f'{n2_str} = {n2[n]:.2e} {n2_u},\n{isat_str} = {isat[i]:.2e} {isat_u}')
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f'{saving_path}unwrap_phase_n2{number_of_n2}_isat{number_of_isat}_power{input_power:.2f}.png')
    plt.close(fig_phase)