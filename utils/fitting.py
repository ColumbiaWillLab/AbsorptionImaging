"""
General fitting helpers
"""
import functools

import numpy as np


def ravel(func):
    """Decorator that ravels the return value of the decorated function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return np.ravel(func(*args, **kwargs))

    return wrapper


def gaussian_2D(x, y, A, x0, y0, sx, sy, theta=0, z0=0):
    """Takes a meshgrid of x, y and returns the gaussian computed across all values.
    See https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function"""
    cos_sq = np.square(np.cos(theta))
    sin_sq = np.square(np.sin(theta))
    sin2th = np.sin(2 * theta)
    sx_sq = np.square(sx)
    sy_sq = np.square(sy)

    # General 2D Gaussian equation parameters
    a = cos_sq / (2 * sx_sq) + sin_sq / (2 * sy_sq)
    b = sin2th / (4 * sy_sq) - sin2th / (4 * sx_sq)
    c = sin_sq / (2 * sx_sq) + cos_sq / (2 * sy_sq)

    quadratic = (
        a * np.square(x - x0) + 2 * b * (x - x0) * (y - y0) + c * np.square(y - y0)
    )
    return A * np.exp(-quadratic) + z0
