import functools

import numpy as np


def ravel(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return np.ravel(func(*args, **kwargs))

    wrapper._decorator_name = "ravel"
    return wrapper


def gaussian_2D(x, y, A, x0, y0, sx, sy, theta=0, z0=0):
    """Takes a meshgrid of x, y and returns the gaussian computed across all values.
    See https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function for more info"""
    cos_sq = np.power(np.cos(theta), 2)
    sin_sq = np.power(np.sin(theta), 2)
    sin2th = np.sin(2 * theta)
    sx_sq = np.power(sx, 2)
    sy_sq = np.power(sy, 2)

    # General 2D Gaussian equation parameters
    a = cos_sq / (2 * sx_sq) + sin_sq / (2 * sy_sq)
    b = sin2th / (4 * sy_sq) - sin2th / (4 * sx_sq)
    c = sin_sq / (2 * sx_sq) + cos_sq / (2 * sy_sq)

    quadratic = (
        a * np.power(x - x0, 2) + 2 * b * (x - x0) * (y - y0) + c * np.power(y - y0, 2)
    )
    return A * np.exp(-quadratic) + z0
