import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_images(
        data: np.ndarray, 
        saving_path: str, 
        nlse_settings: tuple
        ) -> None:
    
    n2, input_power, alpha, isat, waist_input_beam, non_locality_length, delta_z, cell_length = nlse_settings
    number_of_n2 = len(n2)
    number_of_isat = len(isat)
    number_of_alpha = len(alpha)

    field = data.copy().reshape(number_of_n2, number_of_isat, number_of_alpha, 3, data.shape[-2], data.shape[-2])
    density_channels = field[:,  :, :, 0, :, :]
    phase_channels = field[:, :, :, 1, :, :]
    uphase_channels = field[:, :, :, 2, :, :]
    
    n2_str = r"$n_2$"
    n2_u = r"$m^2$/$W$"
    isat_str = r"$I_{sat}$"
    isat_u = r"$W$/$m^2$"
    puiss_str = r"$p$"
    puiss_u = r"$W$"
    alpha_str = r"$\alpha$"
    alpha_u = r"$m^{-1}$"

    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 50

    for alpha_index, alpha_value in enumerate(alpha):
        
        fig_density, axes_density = plt.subplots(number_of_n2, number_of_isat, figsize=(10*number_of_isat,10*number_of_n2))
        fig_density.suptitle(f'Density Channels - {puiss_str} = {input_power:.2e} {puiss_u} - {alpha_str} = {alpha_value:.2e} {alpha_u}')

        for n2_index, n2_value in enumerate(n2):
            for isat_index, isat_value in enumerate(isat):
                ax = axes_density if number_of_n2 == 1 and number_of_isat == 1 else (axes_density[n2_index, isat_index] if number_of_n2 > 1 and number_of_n2 > 1 else (axes_density[n2_index] if number_of_n2 > 1 else axes_density[isat_index]))
                ax.imshow(density_channels[n2_index, isat_index, alpha_index, :, :], cmap='viridis')
                ax.set_title(f'{n2_str} = {n2_value:.2e} {n2_u},\n{isat_str} = {isat_value:.2e} {isat_u}')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{saving_path}/density_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_{alpha_value}_power{input_power:.2f}.png')
        plt.close(fig_density) 

        # Plot phase channels for the current power value
        fig_phase, axes_phase = plt.subplots(number_of_n2, number_of_isat, figsize=(10*number_of_isat,10*number_of_n2))
        fig_phase.suptitle(f'Phase Channels - {puiss_str} = {input_power:.2e} {puiss_u} - {alpha_str} = {alpha_value:.2e} {alpha_u}')

        for n2_index, n2_value in enumerate(n2):
            for isat_index, isat_value in enumerate(isat):
                ax = axes_phase if number_of_n2 == 1 and number_of_isat == 1 else (axes_phase[n2_index, isat_index] if number_of_n2 > 1 and number_of_isat > 1 else (axes_phase[n2_index] if number_of_n2 > 1 else axes_phase[isat_index]))
                ax.imshow(phase_channels[n2_index, isat_index, alpha_index, :, :], cmap='twilight_shifted')
                ax.set_title(f'{n2_str} = {n2_value:.2e} {n2_u},\n{isat_str} = {isat_value:.2e} {isat_u}')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{saving_path}/phase_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_{alpha_value}_power{input_power:.2f}.png')
        plt.close(fig_phase)

        #Plot phase channels for the current power value
        fig_phase, axes_phase = plt.subplots(number_of_n2, number_of_isat, figsize=(50,50))
        fig_phase.suptitle(f'Unwrap Phase Channels - {puiss_str} = {input_power:.2e} {puiss_u} - {alpha_str} = {alpha_value:.2e} {alpha_u}')

        for n2_index, n2_value in enumerate(n2):
            for isat_index, isat_value in enumerate(isat):
                ax = axes_phase if number_of_n2 == 1 and number_of_isat == 1 else (axes_phase[n2_index, isat_index] if number_of_n2 > 1 and number_of_isat > 1 else (axes_phase[n2_index] if number_of_n2 > 1 else axes_phase[isat_index]))
                ax.imshow(uphase_channels[n2_index, isat_index, alpha_index, :, :], cmap='viridis')
                ax.set_title(f'{n2_str} = {n2_value:.2e} {n2_u},\n{isat_str} = {isat_value:.2e} {isat_u}')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{saving_path}/unwrap_phase_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_{alpha_value}_power{input_power:.2f}.png')
        plt.close(fig_phase)