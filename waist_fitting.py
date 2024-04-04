from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the simplified 2D Gaussian function
def simplified_gaussian(xy, sigma, A, x_0, y_0 ):
    x, y = xy
    return A * np.exp(-(((x-x_0)**2 + (y-y_0)**2) / (sigma**2)))

def waist_computation(field, window, NX, NY, plot, path=None):
    X, delta_X = np.linspace(
        -window / 2,
        window / 2,
        num=NX,
        endpoint=False,
        retstep=True,
        dtype=np.float32,
    )
    Y, delta_Y = np.linspace(
        -window / 2,
        window / 2,
        num=NY,
        endpoint=False,
        retstep=True,
        dtype=np.float32,
    )

    XX, YY = np.meshgrid(X, Y)
    XY_ravel =  np.vstack((XX.ravel(), YY.ravel()))
    field_ravel = field.ravel()
    sigma_opt, _ = curve_fit(simplified_gaussian, XY_ravel, field_ravel,p0=[5e-4, 1, 0, 0] )

    sigma_waist = sigma_opt[0]

    if plot:
        E = np.ones((NX, NY), dtype=np.float32) * np.exp(-(XX**2 + YY**2) / (sigma_waist**2))
        plt.title(f"waist = {sigma_waist}")
        plt.imshow(E)
        plt.savefig(f"{path}/gaussian_fitting.png")
        plt.close()
    return sigma_waist