"""Gaussian fitting"""

import cv2
import numpy as np

import Helper as hp
from .utils import gaussian


def two_D_gaussian(mode, f, shot, n):
    print("Mode:", mode)

    transmission = shot.transmission
    width = shot.width
    height = shot.height
    x, y = shot.meshgrid

    # compute parameters automatically
    if mode == "automatic":
        # coarsen the image; create a coarse meshgrid for plotting
        coarse = transmission[::f, ::f]
        x_c = np.linspace(f, len(coarse[0]) * f, len(coarse[0]))
        y_c = np.linspace(f, len(coarse) * f, len(coarse))
        (x_c, y_c) = np.meshgrid(x_c, y_c)

        # take an "intelligent" guess and run the coarse fit
        (y0, x0, peak) = hp.peak_find(coarse, f)  # guess an initial center point
        (amp, z0) = (transmission[0][0] - peak, 1 - transmission[0][0])
        guess = [amp, x0, y0, 200, 200, 0, z0]
        coarse_fit, best = hp.iterfit(
            hp.residual, guess, x_c, y_c, width, height, coarse, n
        )

        # compute the relative error from the coarse fit
        error = (coarse - coarse_fit) / coarse
        area = (width * height) / (f ** 2)
        int_error = (np.sum((error) ** 2) / area) * 1000
        print("Integrated error: " + str(round(int_error, 2)))

    # guess parameters based on user input
    elif mode == "manual":
        # allow the user to select a region of interest
        r = cv2.selectROI(transmission)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # zoom in, coarsen, and create a coarse meshgrid
        zoomed = hp.zoom_in(transmission, r)
        coarse = zoomed[::f, ::f]
        x_c = np.linspace(f, len(coarse[0]) * f, len(coarse[0]))
        y_c = np.linspace(f, len(coarse) * f, len(coarse))
        (x_c, y_c) = np.meshgrid(x_c, y_c)

        # take an intelligent guess at fit parameters
        (y0, x0, peak) = hp.peak_find(coarse, f)
        (amp, z0) = (transmission[0][0] - peak, 1 - transmission[0][0])
        sigma_x = 0.5 * (r[2] / f)
        sigma_y = 0.5 * (r[3] / f)
        guess = [amp, x0, y0, sigma_x, sigma_y, 0, z0]

        # run the zoomed-in fit and compute its relative error
        fine_fit, best = hp.iterfit(
            hp.residual, guess, x_c, y_c, width, height, coarse, n
        )
        best[1] = best[1] + r[0]
        best[2] = best[2] + r[1]
        error = (coarse - fine_fit) / coarse
        area = (r[2] * r[3]) / (f ** 2)
        int_error = (np.sum((error) ** 2) / area) * 1000
        print("Integrated error: " + str(round(int_error, 2)))

    # generate final-fit transmission data; compute relative error

    fit_data = 1 - gaussian(x, y, best)
    final_error = (transmission - fit_data) / transmission

    return final_error, best, None, int_error


def one_D_gaussian(shot, best):
    width = shot.width
    height = shot.height
    transmission = shot.transmission

    # define the best-fit axes
    x_val = np.linspace(-2 * width, 2 * width, 4 * width)
    y_val = np.linspace(-2 * height, 2 * height, 4 * height)
    (x_hor, y_hor, x_ver, y_ver) = hp.lines(x_val, best)

    # collect (Gaussian) data along these axes
    print("Collecting 1D data")
    (x_axis, horizontal) = hp.collect_data(transmission, x_hor, y_hor, "x")
    (y_axis, vertical) = hp.collect_data(transmission, x_ver, y_ver, "y")

    # perform a 1D Gaussian fit on each data set:
    # for the 1D fits, take the guess [A, x0/y0, sigma_x/sigma_y, z0]
    guess_h = np.array([best[0], best[1], best[3], best[6]])
    guess_v = np.array([best[0], best[2], best[4], best[6]])

    # perform the horizontal and vertical 1D fits
    fit_h, param_h = hp.fit_1d(hp.residual_1d, guess_h, x_axis, horizontal)
    fit_v, param_v = hp.fit_1d(hp.residual_1d, guess_v, y_axis, vertical)

    return (
        fit_h,
        fit_v,
        param_h,
        param_v,
        x_hor,
        y_hor,
        x_ver,
        y_ver,
        x_axis,
        y_axis,
        horizontal,
        vertical,
    )
