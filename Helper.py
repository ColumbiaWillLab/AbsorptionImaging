import time

import math
import numpy as np

from scipy.optimize import curve_fit
from scipy.ndimage import filters
from lmfit import minimize, Parameters

import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.gridspec as gridspec

class MyExcept(Exception):
    """
    Raise this when 3 new files have been detected.
    """

# Preliminary data processing
def subtraction(data, beam, dark, kernel):
    """
    INPUT: 3 image arrays (background, laser beam, absorption image).
    PROCESSING: subtract background from both data and beam arrays; divide
    absorption data by beam background to get the transmission t^2.
    Then apply a Gaussian kernel filter to smooth out remaining noise.
    OUTPUT: numpy array containing transmission (0 < t^2 < 1) values.
    """
    print("Performing background subtraction")
    background = beam.astype(int) - dark.astype(int)
    image = data.astype(int) - dark.astype(int)
    transmission = image.astype(float) / background.astype(float)
    time.sleep(.01)
    transmission[background <= 3] = 1
    time.sleep(.01)

    print("Applying Gaussian filter of size " + str(kernel))
    transmission = filters.gaussian_filter(transmission, kernel)
    return transmission

# Auxiliary analysis functions
def params2list(parameters):
    # Converts a Parameters object (parameters) to a list (array)

    fancy = parameters.valuesdict().items()
    values = []
    for entry in fancy:
        values.append(entry[1])
    return values
def list2params(value):
    # Converts a list (val) to a Parameters object (params) 

    params = Parameters()
    params.add('A', value = value[0])
    params.add('x0', value = value[1])
    params.add('y0', value = value[2])
    params.add('sigma_x', value = value[3])
    params.add('sigma_y', value = value[4])
    params.add('theta', value = value[5])
    params.add('z0', value = value[6])
    return params
def peak_find(data, f):
    """
    INPUT: data contains the array searched for a peak.
    f is the factor by which to reduce the resolution.
    OUTPUT: the indices and value (x0, y0, val) of the peak in data.
    """
    # use a peak-finding algorithm like scipy.peakfind.cwt???

    print("Finding a transmission peak")
    flat_data = data.ravel()
    minimum = np.argmin(flat_data)
    shape = (len(data), len(data[0]))
    (y0_ind, x0_ind) = np.unravel_index(minimum, shape)
    val = data[y0_ind][x0_ind]
    (y0, x0) = (x0_ind * f, y0_ind * f)
    print(x0,y0,val)
    return (x0, y0, val)
def de_enhance(data, f):
    """
    INPUT: data contains the high-res image to be blockified;
    f is the factor by which to reduce the resolution.
    OUTPUT: a de-enhanced array containing the image data.
    """

    print("De-enhancing by a factor of " + str(f))
    coarse = []
    w = len(data[0])/f
    h = len(data)/f

    # skip every f pixels and append to a smaller array
    for i in range(h):
        row = []
        for j in range(w):
            row.append(data[i*f][j*f])
        coarse.append(row)
    return np.array(coarse)
def zoom_in(data, r):
    """
    INPUT: data is the large array, zoomed into user-defined ROI 
    given by r = (xmin, ymin, w, h).
    OUTPUT: smaller array zooming in on a Gaussian feature.
    """
    
    # define bounds of zoomed array
    xmin = r[0]
    ymin = r[1]
    xmax = r[0] + r[2]
    ymax = r[1] + r[3]

    # build the zoomed-in array
    zoomed = []
    for row in data[ymin:ymax]:
        new_row = row[xmin:xmax]
        zoomed.append(new_row)

    print("Zooming in: (" +str(r[2])+" p) x ("+str(r[3])+" p)")
    return zoomed

# Gaussian fitting procedure
def gaussian(params, x,y):
    """
    INPUT: params = [A, x0, y0, sigma_x, sigma_y, theta] contains the
    amplitude, center point, standard deviations, and angle of the blob.
    (x,y) is the meshgrid defining the function domain.
    OUTPUT: values of the Gaussian function f(x).
    """

    # Unpack parameters
    values = params2list(params)
    (A, x0, y0, sx, sy, th, z0) = values

    # Define constants (see Wikipedia)
    a = (math.cos(th)**2)/(sx**2) + (math.sin(th)**2)/(sy**2)
    b = -(np.sin(2*th))/(2*sx**2) + (np.sin(2*th))/(2*sy**2)
    c = (np.sin(th)**2)/(sx**2) + (np.cos(th)**2)/(sy**2)

    # Create the Gaussian function
    quadratic = a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2
    return A * np.exp(-0.5*quadratic) + z0
def residual(params, x,y, data):
    """
    INPUT: params = [A, x0, y0, sigma_x, sigma_y, theta, z0].
    (x,y) defines the Gaussian's domain; data is the data to be fitted.
    OUTPUT: flattened error signal array, (1 - gaussian) - data.
    """

    model = 1 - gaussian(params, x,y)
    return (model - data).flatten()
def minimizer(residual, params, x,y, data):
    """
    INPUT: residual is the function to be minimized;
    params contains the parameters to be optimized;
    data contains the data to be fitted to the model.
    OUTPUT: best-fit parameters and set of data based on them.
    """

    # Perform the minimization
    out = minimize(residual, params, xtol = 1e-3, args = (x,y, data))
    best = params2list(out.params)
    param0 = list2params(best)

    # Generate best-fit data
    fit_data = 1 - gaussian(param0, x,y)
    return fit_data, best
def iterfit(residual, guess, x,y, width,height, data, num):
    """
    INPUT: residual is the function (model - data) to be minimized;
    guess contains the starting parameters for fitting by minimizer;
    data is the data to be fitted; num = # of fitting iterations.
    OUTPUT: best-fit parameters and a set of data based on them.
    """

    print("Performing a 2D Gaussian fit")
    # Set initial guess
    p = guess
    print("Guess:    " + str(np.round(p,2)))

    for i in range(num): # iterate the fit [num] times
        # Load parameter object with values from most recent fit
        params = Parameters()
        params.add('A', value = p[0], min = 0, max = 2)
        params.add('x0', value = p[1], min = 0, max = width)
        params.add('y0', value = p[2], min = 0, max = height)
        params.add('sigma_x', value = p[3], min = 1, max = width)
        params.add('sigma_y', value = p[4], min = 1, max = height)
        params.add('theta', value = p[5], min = -math.pi/4, max = math.pi/4)
        params.add('z0', value = p[6], min = -2, max = 2)

        # Do the minimization; redefine the initial guess
        fit_data, p = minimizer(residual, params, x,y, data)

    print("Best fit: " + str(np.round(p,2)))
    return fit_data, p

# 1D analysis and fitting
def lines(x, params):
    """
    INPUT: lists of domain values and the best-fit Gaussian parameters.
    OUTPUT: two lists of coordinates (x_hor, y_hor) and (x_ver, y_ver)
    lying along the horizontal & vertical axes of the Gaussian.
    """

    # Unpack parameters
    (x0, y0) = (params[1], params[2])
    theta = params[5]
    theta = 0

    # point-slope and y = mx + b
    m = -np.tan(theta)
    b = -m*x0 + y0
    y_hor = m*x + b
    x_hor = x

    # rotate the lines
    x_ver = -(y_hor - y0) + x0
    y_ver = (x_hor - x0) + y0

    return x_hor, y_hor, x_ver, y_ver
def collect_data(data, x_val, y_val, axis):
    """
    INPUT: data is the array to be mined for data; x_val and y_val
    are (x,y) locations in pixels where data is to be retreived;
    axis is x_val or y_val, and is the plotting axis for data.
    OUTPUT: a list of coordinates (either x or y) and corresponding
    pixel values along those coordinates.
    """

    # approximate pixel locations of sampled points
    x_pix = np.round(x_val).astype(int)
    y_pix = np.round(y_val).astype(int)
    width = len(data[0])
    height = len(data)

    output_axis = []
    output_data = []
    if axis == 'x':
        ax = x_pix
    elif axis == 'y':
        ax = y_pix

    for i in range(len(ax)): # pick pixel locations in valid range
        if (0 <= y_pix[i] < height) and (0 <= x_pix[i] < width):
            coord = ax[i]
            value = data[y_pix[i]][x_pix[i]]
            output_axis.append(coord)
            output_data.append(value)
    return np.array(output_axis), np.array(output_data)
def gaussian_1d(params, x):
    """
    INPUT: x0, sigma, and z0 are parameters of the Gaussian f(x);
    x is the list containing its domain.
    OUTPUT: a list containing values of the Gaussian 1 - f(x).
    """

    # Unpack parameters
    values = params2list(params)
    (A, x0, sigma, z0) = values

    # Create the Gaussian function
    quadratic = (x-x0)**2 / sigma**2
    return A * np.exp(-0.5*quadratic) + z0
def residual_1d(params, x, data):
    """
    INPUT: params = [A, x0, sigma, z0] are the Gaussian's parameters;
    x is the Gaussian's domain; data is the data to be fitted.
    OUTPUT: error signal list, (1 - gaussian_1d) - data.
    """

    model = 1 - gaussian_1d(params, x)
    return model - data
def fit_1d(residual, guess, x, data):
    """
    INPUT: residual is the function (model - data) to be minimized;
    guess contains the starting parameters for the fit; x is the domain.
    OUTPUT: best-fit parameters and a set of data based on them.
    """

    print("Performing a 1D Gaussian fit")
    p = guess
    print("Guess:    " + str(np.round(p,2)))
    # Set initial guess
    params = Parameters()
    params.add('A', value = p[0], min = 0, max = 2)
    params.add('x0', value = p[1], min = p[1] - 100, max = p[1] + 100)
    params.add('sigma', value = p[2], min = 1, max = len(data))
    params.add('z0', value = p[3], min = -2, max = 2)

    # Do the minimization; redefine the initial guess
    out = minimize(residual, params, xtol = 1e-3, args = (x, data))
    best = params2list(out.params)
    
    # Convert best to a Parameters() object
    # This is shameful programming...
    param0 = Parameters()
    param0.add('A', value = best[0])
    param0.add('x0', value = best[1])
    param0.add('sigma', value = best[2])
    param0.add('z0', value = best[3])

    # Generate best-fit data
    print("Best fit: " + str(np.round(best, 2)))
    fit_data = 1 - gaussian_1d(param0, x)
    return fit_data, best