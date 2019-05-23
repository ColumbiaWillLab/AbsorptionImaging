import numpy as np


def gaussian(x, y, *params):
    """
    INPUT: params = [A, x0, y0, sigma_x, sigma_y, theta] contains the
    amplitude, center point, standard deviations, and angle of the blob.
    (x,y) is the meshgrid defining the function domain.
    OUTPUT: values of the Gaussian function f(x).
    """

    # Unpack parameters
    (A, x0, y0, sx, sy, th, z0) = params[0]

    # Define constants (see Wikipedia)
    a = (np.cos(th) ** 2) / (sx ** 2) + (np.sin(th) ** 2) / (sy ** 2)
    b = -(np.sin(2 * th)) / (2 * sx ** 2) + (np.sin(2 * th)) / (2 * sy ** 2)
    c = (np.sin(th) ** 2) / (sx ** 2) + (np.cos(th) ** 2) / (sy ** 2)

    # Create the Gaussian function
    quadratic = a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2
    return A * np.exp(-0.5 * quadratic) + z0
