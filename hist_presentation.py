import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 20

#no seed
# n2 =   np.array([-1.627952605485916e-9, -1.8238249421119688e-9, -1.5391492843627928e-09, -1.588274836540222e-09, -1.4611566066741941e-09, -1.3521094620227811e-09, -1.2878944035735968e-09])
# Isat = np.array([81793.70760917664    , 69090.44981002808     , 43737.62607574463      , 50193.08924674988     , 51781.731843948364     , 50684.72623825073      , 76842.93830580918])

#seed 0
# n2 =   np.array([-1.627952605485916e-9, -1.309995949268341e-09, -1.6505193710327146e-09, -1.7480167746543882e-09, None, -1.469030976295471e-09, -1.2878944035735968e-09])
# Isat = np.array([81793.70760917664    , 38750.0524520874     , 1821.2884664535522      , 63123.464584350586     , None, 53609.32946205139      , 76842.93830580918])

#seed 10
n2 = np.array([  -2.013047039508819e-09, -2.194893658161163e-09, -1.7534224689006802e-09, None, None, -2.1258921921253203e-09, -1.2878944035735968e-09])
Isat = np.array([82905.49516677856     , 52120.41735649109     , 47410.70866584778      , None, None, 41584.26225185394      , 76842.93830580918])
n2_range = [-5e-9, -5e-11]
Isat_range = [1e4, 1e6]
n2Isat_range = [-1e-6, -1e-4]
xticks_counts = [5, 10, 20, 30, 40, 50, 50]

n2_str = r"$n_2$"
n2_u = r"$m^2$/$W$"
isat_str = r"$I_{sat}$"
isat_u = r"$W$/$m^2$"
puiss_str = r"$p$"
puiss_u = r"$W$"

# n2 points: 4 subplots vertically aligned with consistent range and rotated ticks
fig_n2, axs_n2 = plt.subplots(len(n2), 1, figsize=(20, len(n2)*3))  # Further increased figsize
for i, ax in enumerate(axs_n2):
    if i != len(n2) - 1:
        ax.set_title(f'For {xticks_counts[i]} n2 and {xticks_counts[i]} Isat: {n2_str} = {n2[i]:.2e} {n2_u}')
    else:
        ax.set_title(f'Tangui: {n2_str} = {n2[i]:.2e} {n2_u}')

    ax.plot(n2[i], 0, 'b^', markersize=30) 
    ax.set_xlim(n2_range)
    ax.set_xlabel(f"{n2_str} ({n2_u})")
    ax.set_xticks(np.linspace(n2_range[0], n2_range[1], xticks_counts[i]))
    ax.grid(True)
    ax.set_yticks([])
    ax.set_xticklabels([f'{tick:.2e}' for tick in np.linspace(n2_range[0], n2_range[1], xticks_counts[i])], rotation=45)
    

fig_n2.tight_layout()
plt.savefig("/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/distributions_n2.png")
# Isat points: 4 subplots vertically aligned with consistent range and rotated ticks
fig_Isat, axs_Isat = plt.subplots(len(Isat), 1, figsize=(20, len(Isat)*3))  # Further increased figsize
for i, ax in enumerate(axs_Isat):
    if i != len(Isat) - 1:
        ax.set_title(f'For {xticks_counts[i]} n2 and {xticks_counts[i]} Isat: {isat_str} = {Isat[i]:.2e} {isat_u}')
    else:
        ax.set_title(f'Tangui: {isat_str} = {Isat[i]:.2e} {isat_u}')
        
    ax.plot(Isat[i], 0, 'ro', markersize=30)  # Plot each Isat value in its own subplot
    ax.set_xlim(Isat_range)
    ax.set_xlabel(f"{isat_str} ({isat_u})")
    ax.set_xticks(np.linspace(Isat_range[0], Isat_range[1], xticks_counts[i]))
    ax.grid(True)
    ax.set_yticks([])
    ax.set_xticklabels([f'{tick:.2e}' for tick in np.linspace(Isat_range[0], Isat_range[1], xticks_counts[i])], rotation=45)

fig_Isat.tight_layout()
plt.savefig("/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/distributions_isat.png")