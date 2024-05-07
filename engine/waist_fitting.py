#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def gaussian(xy, sigma, A, x_0, y_0 ):
    """
    Computes the value of a 2D Gaussian function at given coordinates.

    This function models a 2D Gaussian distribution with a specified amplitude, center,
    and standard deviation (which is related to the 'waist' in physics contexts). It's 
    commonly used in optics and image processing to represent beam profiles, blur effects, 
    or spatial distributions.

    Parameters:
    - xy (tuple of np.ndarray): A tuple containing two arrays, x and y coordinates, where the 
      Gaussian function is to be evaluated. The shapes of these arrays should be compatible 
      for broadcasting in NumPy.
    - sigma (float): The standard deviation of the Gaussian distribution. In optics, this 
      parameter is often related to the beam waist, determining the spread of the beam.
    - A (float): The amplitude of the Gaussian peak. This determines the maximum value of 
      the Gaussian function, typically representing the peak intensity of a beam.
    - x_0 (float): The x-coordinate of the center of the Gaussian peak.
    - y_0 (float): The y-coordinate of the center of the Gaussian peak.

    Returns:
    - np.ndarray: The values of the Gaussian function evaluated at the specified x and y 
      coordinates. This will have the same shape as the input coordinate arrays, representing 
      the intensity or value of the Gaussian distribution at each point.

    Example:
    To compute the Gaussian values on a 5x5 grid centered at (0,0) with a standard deviation 
    of 1 and an amplitude of 1:

        x = np.linspace(-2, 2, 5)
        y = np.linspace(-2, 2, 5)
        X, Y = np.meshgrid(x, y)
        Z = gaussian((X, Y), 1, 1, 0, 0)
    
    This example would produce a 2D array (Z) representing the Gaussian values across the grid.
    """
    x, y = xy
    return A * np.exp(-(((x-x_0)**2 + (y-y_0)**2) / (sigma**2)))

def pinhole(
    field: np.ndarray,
    window: float, 
    NX: int, 
    NY: int, 
    plot: bool, 
    pinsize: float,
    ) -> float:
    """
    Computes the waist of a beam by fitting a simplified 2D Gaussian function to a given field. 
    Optionally, it can plot and save an image of the Gaussian fit.

    Parameters:
    - field (np.ndarray): A 2D numpy array representing the input field to be analyzed. 
      The function attempts to fit a Gaussian to this field.
    - window (float): The size of the square window in meters across which the field is defined. 
      Used to determine the spatial coordinates over which the field is sampled.
    - NX (int): The number of points in the X dimension for creating the meshgrid over which 
      the Gaussian is fitted.
    - NY (int): The number of points in the Y dimension for creating the meshgrid.
    - plot (bool): If True, the function will plot the Gaussian fit and the computed waist 
      over the meshgrid defined by NX and NY. The plot is saved if a path is provided.
    - path (str, optional): The directory path to save the plot of the Gaussian fit. Required 
      if `plot` is True. The file is saved as 'gaussian_fitting.png'.

    Returns:
    - sigma_waist (float): The computed waist of the beam, derived from the optimal fit of the 
      simplified Gaussian model to the input field. This value represents the standard deviation 
      of the Gaussian, which is related to the beam waist.

    This function uses the `curve_fit` method from `scipy.optimize` to fit a simplified 2D Gaussian 
    function to the input field. The Gaussian is defined by a standard deviation (sigma), amplitude (A), 
    and its center (x_0, y_0). The waist of the beam is equivalent to the fitted sigma value. If plotting 
    is enabled and a path is provided, it visualizes the fit and saves the plot as an image.
    """
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
    sigma_opt, _ = curve_fit(gaussian, XY_ravel, field_ravel,p0=[5e-4, 1, 0, 0] )

    sigma_waist = sigma_opt[0]
    R = np.hypot(XX, YY)
    field[R > pinsize*sigma_waist] = 0

    if plot:
        E = np.ones((NX, NY), dtype=np.float32) * np.exp(-(XX**2 + YY**2) / (sigma_waist**2))
        plt.title(f"waist = {sigma_waist}")
        plt.imshow(E)
        plt.savefig(f"gaussian_fitting.png")
        plt.close()

        plt.title(f"waist = {sigma_waist}")
        plt.imshow(field)
        plt.savefig(f"gaussian_pinhole.png")
        plt.close()
    return field