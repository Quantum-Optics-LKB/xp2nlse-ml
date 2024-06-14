from NLSE import NLSE
import numpy as np
import pyfftw
from scipy.constants import c, epsilon_0

from engine.nlse_generator import add_model_noise

PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32


N = 2048
n2 = -1.3e-09
Isat = 1e6
waist = 2.3e-3
window = 4*waist
power = 1.05
L = 20e-2
alpha = 22
nl_length = 0#30e-6
dz = 1e-5



simu = NLSE(alpha, power, window, n2, None, L, NX=N, NY=N, Isat=Isat)
simu.nl_length = nl_length
simu.delta_z = dz
beam = np.ones((N, N), dtype=np.complex64)*np.exp(-(simu.XX**2 + simu.YY**2) / waist**2)
poisson_noise_lam, normal_noise_sigma = 0.1 , 0.01
beam = add_model_noise(beam, poisson_noise_lam, normal_noise_sigma)
A = simu.out_field(beam, L, plot=True, verbose=True, normalize=True)