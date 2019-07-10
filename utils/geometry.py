"""General geometry helpers"""
import numpy as np


def clipped_endpoints(a, a_c, b_c, m, b_max):
    """Calculates the coordinates of the line endpoint at the independent variable a,
    given a center (a_c, b_c) and a slope m. If the dependent variable lies outside
    the bounds of (0, b_max), then we clip and return the correct bounded endpoint."""
    b = b_c + m * (a_c - a)  # i.e. y = y_0 + m * (x_0 - x)

    if b < 0:
        b = 0
        a = b_c / m + a_c
    elif b > b_max:
        b = b_max
        a = (b_c - b) / m + a_c

    return tuple(np.floor([a, b]).astype(int))
