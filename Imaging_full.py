##################################################
######## 0: CLASSES, FUNCTIONS, LIBRARIES ########
##################################################

print("0: SET UP LIBRARIES/FUNCTIONS")
print("------------------------------")

import time
start = time.clock()

import os
import shutil
import sys
import logging
from watchdog.observers import Observer
import imageio

import math
import numpy as np
np.set_printoptions(suppress=True) # suppress scientific notation

from scipy.optimize import curve_fit
from scipy.ndimage import filters
from lmfit import minimize, Parameters

import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import matplotlib.gridspec as gridspec

# mode = 'manual' or 'automatic'
mode = 'automatic'

# all function definitions and classes
class MyExcept(Exception):
    """
    Raise this when 3 new files have been detected.
    """

class FileSystemEventHandler(object):
    """
    Base file system event handler that you can override methods from.
    Necessary but not relevant, so don't modify.
    """
    def dispatch(self, event):
        """Dispatches events to the appropriate methods.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`FileSystemEvent`
        """
        self.on_any_event(event)
        _method_map = {
            EVENT_TYPE_MODIFIED: self.on_modified,
            EVENT_TYPE_MOVED: self.on_moved,
            EVENT_TYPE_CREATED: self.on_created,
            EVENT_TYPE_DELETED: self.on_deleted,
        }
        event_type = event.event_type
        _method_map[event_type](event)
    def on_any_event(self, event):
        """Catch-all event handler.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`FileSystemEvent`
        """
    def on_moved(self, event):
        """Called when a file or a directory is moved or renamed.

        :param event:
            Event representing file/directory movement.
        :type event:
            :class:`DirMovedEvent` or :class:`FileMovedEvent`
        """
    def on_created(self, event):
        """Called when a file or directory is created.

        :param event:
            Event representing file/directory creation.
        :type event:
            :class:`DirCreatedEvent` or :class:`FileCreatedEvent`
        """
    def on_deleted(self, event):
        """Called when a file or directory is deleted.

        :param event:
            Event representing file/directory deletion.
        :type event:
            :class:`DirDeletedEvent` or :class:`FileDeletedEvent`
        """
    def on_modified(self, event):
        """Called when a file or directory is modified.

        :param event:
            Event representing file/directory modification.
        :type event:
            :class:`DirModifiedEvent` or :class:`FileModifiedEvent`
        """
class LoggingEventHandler(FileSystemEventHandler):
    """
    Logs all filesystem events and counts the number of file creations.
    For our purposes only on_created and on_deleted are relevant.

    def on_moved(self, event):
        global count
        super(LoggingEventHandler, self).on_moved(event)

        # what = 'directory' if event.is_directory else 'file'
        # filename = event.src_path
        # count += 1
        # print("Moved  " + filename + "; count = " + str(count))
    """
    def on_created(self, event):
        # Counts created files, and stores their filenames.

        global count # counter starts at 0
        global despacito # empty array
        super(LoggingEventHandler, self).on_created(event)
        what = 'directory' if event.is_directory else 'file'
        filename = event.src_path

        despacito.append(filename)
        count += 1 # tick the counter for each new file
        print("Created " + filename + "; count = " + str(count))
        return despacito

        # execute arbitrary code!
    def on_deleted(self, event):
        global count
        super(LoggingEventHandler, self).on_deleted(event)

        what = 'directory' if event.is_directory else 'file'
        filename = event.src_path
        if despacito != []:
            del despacito[-1]
        else:
            count += 1
        count -= 1
        print("Deleted " + filename + "; count = " + str(count))
    """
    def on_modified(self, event):
        global count
        super(LoggingEventHandler, self).on_modified(event)

        # what = 'directory' if event.is_directory else 'file'
        # filename = event.src_path
        # count += 1
        # print("Modified " + filename + "; count = " + str(count))

    def on_any_event(self, event):
        global count
        super(LoggingEventHandler, self).on_modified(event)

        what = 'directory' if event.is_directory else 'file'
        filename = event.src_path
        count += 1
        print("Something happened to "+filename+"; count = "+str(count))
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
    transmission[background == 0] = 1

    print("Applying Gaussian filter of size " + str(kernel))
    transmission = filters.gaussian_filter(transmission, kernel)
    return transmission
def fake_data(laser, atoms, noise):
    """
    INPUT: laser and atoms contain Gaussian parameters for the laser beam
    and atomic absorption profiles; noise is the size of shot noise.
    OUTPUT: arrays data, beam, dark containing the 3 fake image arrays.
    """
    print("Generating fake Gaussian data")

    # parameters for fake laser beam and atom sample
    param_l = list2params(laser)
    param_a = list2params(atoms)

    # generate laser-beam and atomic-sample gaussians
    laser = gaussian(param_l, x,y)
    atoms = gaussian(param_a, x,y)
    # dirty = noise * np.random.normal(size=(height,width))

    # create noisy fake data arrays
    data = atoms + laser + noise*np.random.normal(size=(height,width))
    beam = laser + noise * np.random.normal(size = (height, width))
    dark = 1 + noise * np.random.normal(size = (height, width))
    return data, beam, dark

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
def peak_find(data):
    """
    INPUT: data contains the array searched for a peak.
    OUTPUT: the indices and value (x0, y0, val) of the peak in data.
    """
    # use a peak-finding algorithm like scipy.peakfind.cwt???

    print("Finding a transmission peak")
    flat_data = data.ravel()
    minimum = np.argmin(flat_data)
    shape = (len(data), len(data[0]))
    (y0_ind, x0_ind) = np.unravel_index(minimum, shape)
    val = data[y0_ind][x0_ind]
    (y0, x0) = (y0_ind * fy, x0_ind * fx)
    return (y0, x0, val)
def de_enhance(data, fx, fy):
    """
    INPUT: data contains the high-res image to be blockified;
    fx and fy are the factors by which to reduce the resolution.
    OUTPUT: a de-enhanced array containing the image data.
    """

    print("De-enhancing by a factor of "+str(fx)+" x "+str(fy))
    coarse = []
    w = len(data[0])/fx
    h = len(data)/fy

    # skip every (fx, fy) pixels and append to a smaller array
    for i in range(h):
        row = []
        for j in range(w):
            row.append(data[i*fy][j*fx])
        coarse.append(row)
    return np.array(coarse)
def zoom_in(data, params, f):
    """
    INPUT: data is the large array, zoomed in around center = (x0, y0)
    and of dimensions = (f*sigma_x) x (f*sigma_y), contained in params.
    manual contains (optional) user-set parameters for the zooming.
    OUTPUT: smaller array zooming in on a Gaussian feature.
    """

    # decide between automatic or manual mode of operation

    # unpack parameters
    (x0, y0, w, h) = np.array(params[1:5])
    # dim = np.array([(x0-f*w), (x0+f*w), (y0+f*h), (y0-f*h)])

    if mode == 'automatic':
        xmin = int(np.max([0, np.round(x0 - f*w)]))
        xmax = int(np.min([width, np.round(x0 + f*w)]))
        ymin = int(np.max([0, np.round(y0 - f*h)]))
        ymax = int(np.min([height, np.round(y0 + f*h)]))
        dim = np.array([xmin, xmax, ymax, ymin])
    elif mode == 'manual':
        dim = np.array([x_min, x_max, y_max, y_min])
        (xmin, xmax, ymax, ymin)  = [int(i) for i in dim]

    print(dim)
    # build the zoomed-in array
    zoomed = []
    for row in data[ymin:ymax]:
        new_row = row[xmin:xmax]
        zoomed.append(new_row)

    x_ext = np.round(dim[1] - dim[0], 2)
    y_ext = np.round(dim[2] - dim[3], 2)
    print("Zooming in: (" +str(x_ext)+" p) x ("+str(y_ext)+" p)")
    return zoomed, dim

# Region of interest class
class roipoly:
    def __init__(self, fig=[], ax=[], roicolor='b'):
        if fig == []:
            fig = plt.gcf()

        if ax == []:
            ax = plt.gca()

        self.previous_point = []
        self.allxpoints = []
        self.allypoints = []
        self.start_point = []
        self.end_point = []
        self.line = None
        self.roicolor = roicolor
        self.fig = fig
        self.ax = ax
        #self.fig.canvas.draw()

        self.__ID1 = self.fig.canvas.mpl_connect(
            'motion_notify_event', self.__motion_notify_callback)
        self.__ID2 = self.fig.canvas.mpl_connect(
            'button_press_event', self.__button_press_callback)

        if sys.flags.interactive:
            plt.show(block=False)
        else:
            plt.show()
    def getMask(self, currentImage):
        ny, nx = np.shape(currentImage)
        poly_verts = [(self.allxpoints[0], self.allypoints[0])]
        for i in range(len(self.allxpoints)-1, -1, -1):
            poly_verts.append((self.allxpoints[i], self.allypoints[i]))

        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T

        ROIpath = mplPath.Path(poly_verts)
        grid = ROIpath.contains_points(points).reshape((ny,nx))

        # generate minimum and maximum x & y coordinates
        x_cand = []
        y_cand = []
        for coordinate in poly_verts:
            (x_val, y_val) = coordinate
            x_cand.append(x_val)
            y_cand.append(y_val)
        (x_min, x_max) = (min(x_cand), max(x_cand))
        (y_min, y_max) = (min(y_cand), max(y_cand))
        coords = [x_min, x_max, y_min, y_max]
        return grid, coords
    def displayROI(self,**linekwargs):
        l = plt.Line2D(self.allxpoints +
                     [self.allxpoints[0]],
                     self.allypoints +
                     [self.allypoints[0]],
                     color=self.roicolor, **linekwargs)
        ax = plt.gca()
        ax.add_line(l)
        plt.draw()
    def displayMean(self,currentImage, **textkwargs):
        mask, coords = self.getMask(currentImage)
        meanval = np.mean(np.extract(mask, currentImage))
        stdval = np.std(np.extract(mask, currentImage))
        string = "%.3f +- %.3f" % (meanval, stdval)
        plt.text(self.allxpoints[0], self.allypoints[0],
                 string, color=self.roicolor,
                 bbox=dict(facecolor='w', alpha=0.6), **textkwargs)
    def __motion_notify_callback(self, event):
        if event.inaxes:
            ax = event.inaxes
            x, y = event.xdata, event.ydata
            if (event.button == None or event.button == 1) and self.line != None:
                self.line.set_data([self.previous_point[0], x],
                                   [self.previous_point[1], y])
                self.fig.canvas.draw()
    def __button_press_callback(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            ax = event.inaxes
            if event.button == 1 and event.dblclick == False:  # single left click
                if self.line == None: # if there is no line, create a line
                    self.line = plt.Line2D([x, x],
                                           [y, y],
                                           marker='o',
                                           color=self.roicolor)
                    self.start_point = [x,y]
                    self.previous_point =  self.start_point
                    self.allxpoints=[x]
                    self.allypoints=[y]

                    ax.add_line(self.line)
                    self.fig.canvas.draw()
                    # add a segment
                else: # if there is a line, create a segment
                    self.line = plt.Line2D([self.previous_point[0], x],
                                           [self.previous_point[1], y],
                                           marker = 'o',color=self.roicolor)
                    self.previous_point = [x,y]
                    self.allxpoints.append(x)
                    self.allypoints.append(y)

                    event.inaxes.add_line(self.line)
                    self.fig.canvas.draw()
            elif ((event.button == 1 and event.dblclick==True) or
                  (event.button == 3 and event.dblclick==False)) and self.line != None:
                self.fig.canvas.mpl_disconnect(self.__ID1) #joerg
                self.fig.canvas.mpl_disconnect(self.__ID2) #joerg

                self.line.set_data([self.previous_point[0],
                                    self.start_point[0]],
                                   [self.previous_point[1],
                                    self.start_point[1]])
                ax.add_line(self.line)
                self.fig.canvas.draw()
                self.line = None

                if sys.flags.interactive:
                    pass
                else:
                    #figure has to be closed so that code can continue
                    plt.close(self.fig)

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
def iterfit(residual, guess, x,y, data, num):
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
    # theta = 0

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
    out = minimize(residual, params, args = (x, data))
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

stop = time.clock()
print("Initializing took " + str(round(stop - start, 2)) + " seconds")
print(" ")

##################################################
######### 1: READ FILES AND CREATE DATA ##########
##################################################

shot = 1

# main data analysis loop; runs until ctrl+c or ctrl+. interrupts
while True:
    initial = time.clock()
    print("SHOT " + str(shot))
    print(" ")

    print("1: READ FILES AND CREATE DATA")
    print("-------------------------------")
    start = time.clock()

    # Watchdog file monitoring
    count = 0
    despacito = []
    EVENT_TYPE_MOVED = 'moved'
    EVENT_TYPE_DELETED = 'deleted'
    EVENT_TYPE_CREATED = 'created'
    EVENT_TYPE_MODIFIED = 'modified'

    # main watchdog loop - counts to 3
    if __name__ == "__main__":
        print("Watching for new files")
        logging.basicConfig(level=logging.INFO)
        path = sys.argv[1] if len(sys.argv) > 1 else '.'
        event_handler = LoggingEventHandler()
        observer = Observer()
        observer.schedule(event_handler, path, recursive=True)
        observer.start()

        try:
            while True:
                time.sleep(1)
                if count >= 3:
                    print("3 new images: watchdog naptime")
                    raise MyExcept()
        except MyExcept:
            observer.stop()
        observer.join()

    # retrieve names of newly created image files
    despacito2 = []
    for meme in despacito:
        meme = meme[2:]
        despacito2.append(meme)

    # read images into large arrays of pixel values
    print("Writing image data into arrays")
    data = imageio.imread(despacito2[0])
    beam = imageio.imread(despacito2[1])
    dark = imageio.imread(despacito2[2])
    width = len(data[0])
    height = len(data)

    # save raw images in a new folder
    garbage_path = './Raw Data/'
    now = time.strftime("%Y%m%d-%H%M%S")
    pic_num = 1

    for meme in despacito2:
        name = "Raw_%s_%s.bmp" % (now, str(pic_num))
        os.rename(meme, name)
        shutil.copy2(name, garbage_path)
        pic_num += 1
        os.remove(name)

    # create a meshgrid: each pixel is (3.75 um) x (3.75 um); the image has
    # a resolution of (964 p) x (1292 p) --> (3.615 mm) x (4.845 mm).
    pixelsize = 3.75e-3 # 3.75 um, reported in mm.
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    (x,y) = np.meshgrid(x,y)
    pixels = [0, pixelsize * width, pixelsize * height, 0]

    # gaussian parameters: p = [A, x0, y0, sigma_x, sigma_y, theta, z0]
    # width = 1292
    # height = 964

    # fake data for debugging
    """
    if shot == 1:
        laser = [250, 500, 400, 400, 300, 0.1*math.pi/2, 10]
        atoms = [-200, 600, 500, 30, 50, -.1*math.pi/2, 0]
    elif shot == 2:
        laser = [250, 800, 600, 300, 400, -.1*math.pi/2, 10]
        atoms = [-200, 700, 700, 20, 30, 0.5*math.pi/2, 0]
    elif shot == 3:
        laser = [250, 400, 700, 200, 300, math.pi/2, 10]
        atoms = [-200, 500, 600, 70, 40, 0, 0]
    """

    # create fake data for laser & atom sample; do background subtraction
    kernel = 2
    noise = 1
    # data, beam, dark = fake_data(laser, atoms, noise)
    transmission = subtraction(data, beam, dark, kernel)
    plt.figure(1)
    plt.imshow(transmission, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.show()

    stop = time.clock()
    print("Writing data took " + str(round(stop - start, 2)) + " seconds")

    print(" ")

    ##################################################
    ######### 2: GAUSSIAN FITTING (2D IMAGE) #########
    ##################################################

    print("2: GAUSSIAN FITTING (2D IMAGE)")
    print("-------------------------------")
    start = time.clock()

    print("Mode: " + mode)

    # compute parameters automatically
    if mode == 'automatic':
        (fx, fy) = (5, 5)

        # coarsen the image; create a coarse meshgrid for plotting
        coarse = de_enhance(transmission, fx, fy)
        x_c = np.linspace(fx, len(coarse[0])*fx, len(coarse[0]))
        y_c = np.linspace(fy, len(coarse) * fy, len(coarse))
        (x_c, y_c) = np.meshgrid(x_c, y_c)

        # take an "intelligent" guess and run the coarse fit
        (y0, x0, peak) = peak_find(coarse) # guess an initial center point
        (amp, z0) = (transmission[0][0] - peak, 1 - transmission[0][0])
        guess = [amp, x0, y0, 200, 200, 0, z0]
        coarse_fit, best = iterfit(residual,guess,x_c, y_c,coarse,1)

        # compute the relative error from the coarse fit
        error = (coarse - coarse_fit) / coarse
        area = (width * height) / (fx * fy)
        int_error = (np.sum((error)**2) / area) * 1000
        print("Integrated error: " + str(round(int_error, 2)))

    # guess parameters based on user input
    elif mode == 'manual':
        (fx, fy) = (2, 2)

        # allow the user to select a region of interest
        plt.figure(1)
        plt.imshow(transmission, cmap = 'inferno', extent = pixels)
        plt.colorbar()
        ROI = roipoly(roicolor = 'b')
        mask, coords = ROI.getMask(transmission)

        # take an "intelligent" guess
        (x_min, x_max, y_min, y_max) = coords
        (y0, x0, peak) = peak_find(transmission)
        (amp, z0) = (transmission[0][0] - peak, 1 - transmission[0][0])
        sigma_x = 0.5*(x_max - x_min)
        sigma_y = 0.5*(y_max - y_min)
        guess = [amp, x0, y0, sigma_x, sigma_y, 0, z0]

        # create a new (coarse) meshgrid for plotting
        zoomed, dim_z = zoom_in(transmission, guess, 4)
        coarse = de_enhance(zoomed, fx, fy)
        x_c = np.linspace(fx, len(coarse[0])*fx, len(coarse[0]))
        y_c = np.linspace(fy, len(coarse) * fy, len(coarse))
        (x_c, y_c) = np.meshgrid(x_c, y_c)

        # run the zoomed-in fit and compute its relative error
        fine_fit, best = iterfit(residual, guess, x_c, y_c, coarse, 1)
        error = (coarse - fine_fit) / coarse
        area = ((x_max - x_min) * (y_max - y_min)) / (fx * fy)
        int_error = (np.sum((error)**2) / area) * 1000
        print("Integrated error: " + str(round(int_error, 2)))

    # generate final-fit transmission data; compute relative error
    params0 = list2params(best)
    fit_data = 1 - gaussian(params0, x,y)
    final_error = (transmission - fit_data) / transmission

    stop = time.clock()
    print("2D fitting took " + str(round(stop - start, 2)) + " seconds")
    print(" ")

    ##################################################
    ######## 3: GAUSSIAN FITTING (1D SLICES) #########
    ##################################################

    print("3: GAUSSIAN FITTING (1D SLICES)")
    print("-------------------------------")
    start = time.clock()

    # define the best-fit axes
    x_val = np.linspace(-2*width, 2*width, 4*width)
    y_val = np.linspace(-2*height, 2*height, 4*height)
    (x_hor, y_hor, x_ver, y_ver) = lines(x_val, best)

    # collect (Gaussian) data along these axes
    print("Collecting 1D data")
    (x_axis, horizontal) = collect_data(transmission, x_hor, y_hor, 'x')
    (y_axis, vertical) = collect_data(transmission, x_ver, y_ver, 'y')

    # perform a 1D Gaussian fit on each data set:
    # for the 1D fits, take the guess [A, x0/y0, sigma_x/sigma_y, z0]
    guess_h = np.array([best[0], best[1], best[3], best[6]])
    guess_v = np.array([best[0], best[2], best[4], best[6]])

    # perform the horizontal and vertical 1D fits
    fit_h, param_h = fit_1d(residual_1d, guess_h, x_axis, horizontal)
    fit_v, param_v = fit_1d(residual_1d, guess_v, y_axis, vertical)

    stop = time.clock()
    print("1D fitting took " + str(round(stop - start, 2)) + " seconds")
    print(" ")

    ##################################################
    ########## 4: PHYSICAL DENSITY ANALYSIS ##########
    ##################################################

    print("4: PHYSICAL DENSITY ANALYSIS")
    print("-------------------------------")
    start = time.clock()

    # sodium and camera parameters
    lam = 589.158e-9 # resonant wavelength
    sigma_0 = (3.0/(2.0*math.pi)) * (lam)**2 # cross-section
    area = (pixelsize * 1e-3)**2 # pixel area in SI units

    density = -np.log(transmission)
    atom_num = (area/sigma_0) * np.sum(density)
    print("Atom number: " + str(np.round(atom_num/1e6, 2)) + " million")

    stop = time.clock()
    print("Doing physics took " + str(round(stop - start, 2)) + " seconds")
    print(" ")

    ##################################################
    ######### 5: GRAPHS, PLOTS, AND PICTURES #########
    ##################################################

    print("5: GRAPHS, PLOTS, AND PICTURES")
    print("-------------------------------")
    start = time.clock()

    # preliminary plots: 3 images and transmission
    """
    plt.figure(1)
    plt.imshow(data, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.figure(2)
    plt.imshow(beam, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.figure(3)
    plt.imshow(dark, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.figure(4)
    plt.imshow(data - dark, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.figure(5)
    plt.imshow(beam - dark, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.figure(6)
    plt.imshow(transmission, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.show()
    """

    # coarse and fine fits, and relative errors
    plt.figure(1)
    plt.imshow(final_error, cmap = 'inferno', extent = pixels)
    plt.colorbar()
    plt.show()

    stop = time.clock()
    print("Plotting graphs took " + str(round(stop - start, 2)) + " seconds")
    print(" ")

    ##################################################
    ######### 6: SAVE AND DISPLAY FINAL PLOT #########
    ##################################################

    # create figure: 3x3 grid (boxes labeled 0-8) containing:
    # - title (1);
    # - best-fit parameter outputs (8)
    # - horizontal (4) and vertical (6) plots
    # - transmission plot, in color (7)
    # - empty boxes: 0, 2, 3, 5

    print("6: SAVE AND DISPLAY FINAL PLOT")
    print("-------------------------------")
    start = time.clock()

    print("Painting Gaussian fits in oil")
    norm_min = -0.2
    norm_max = 0.9
    norm = plt.Normalize(norm_min, norm_max)

    fig = plt.figure(1)
    wr = [0.5, 5, 0.5]
    hr = [0.001, 0.5, 4]
    gs = gridspec.GridSpec(3, 3, width_ratios = wr, height_ratios = hr)
    font = {'size'   : 6}
    plt.rc('font', **font)

    # title and pixel scale
    ax1 = plt.subplot(gs[1])
    plt.axis('off')
    title = 'Gaussian Fit to Absorption Data: shot ' + str(shot)
    plt.text(0.1, 1, title, fontsize = 10)

    # best-fit parameters: convert to text
    A = str(np.round(best[0], 2))
    x_0 = str(np.round(pixelsize * best[1], 3))
    y_0 = str(np.round(pixelsize * best[2], 3))
    w_x = str(np.round(2 * pixelsize * best[3], 3))
    w_y = str(np.round(2 * pixelsize * best[4], 3))
    # w_x = str(np.round(2 * pixelsize * param_h[2], 2))
    # w_y = str(np.round(2 * pixelsize * param_v[2], 2))
    (theta, z_0) = (str(np.round(best[5], 2)), str(np.round(best[6], 2)))

    text1 = 'Best-fit parameters:'
    text2 = 'A = ' + A
    text3 = 'x_0 = ' + x_0
    text4 = 'y_0 = ' + y_0
    text5 = 'w_x = '+ w_x
    text6 = 'w_y = '+ w_y
    # text7 = 'theta = '+ theta + ' rad'
    text8 = 'N = ' + str(np.round(atom_num/1000000.0, 2)) + ' million'

    # best-fit parameters: display
    ax8 = plt.subplot(gs[8])
    plt.axis('off')
    plt.text(0, 0.8, text1)
    plt.text(0, 0.7, text2)
    plt.text(0, 0.6, text3)
    plt.text(0, 0.5, text4)
    plt.text(0, 0.4, text5)
    plt.text(0, 0.3, text6)
    plt.text(0, 0.2, text8)

    # horizontal and vertical 1D fits
    ax4 = plt.subplot(gs[4])
    plt.plot(x_axis, 1 - horizontal, 'ko', markersize = 1)
    plt.plot(x_axis, 1 - fit_h, 'r', linewidth = 0.5)
    plt.xlim(0, width)
    plt.ylim(norm_min, norm_max)
    plt.gca().axes.get_xaxis().set_visible(False)

    ax6 = plt.subplot(gs[6])
    plt.plot(1 - vertical, y_axis, 'ko', markersize = 1)
    plt.plot(1 - fit_v, y_axis, 'r', linewidth = 0.5)
    plt.xlim(norm_max, norm_min)
    plt.ylim(height, 0)
    plt.gca().axes.get_yaxis().set_visible(False)

    # transmission plot with axis lines and zoom box
    ax7 = plt.subplot(gs[7])
    plt.imshow(1 - transmission, cmap='inferno', norm=norm, extent=pixels)
    plt.plot(pixelsize*x_hor, pixelsize*y_hor, color = 'g', linewidth = 0.5)
    plt.plot(pixelsize*x_ver, pixelsize*y_ver, color = 'g', linewidth = 0.5)

    plt.xlim(pixels[0], pixels[1])
    plt.ylim(pixels[2], pixels[3])

    # save best-fit parameters and image to files
    save_path = './'
    now = time.strftime("%Y%m%d-%H%M%S")

    pic_path = save_path + '/Analysis Results/'+ now + '.png'
    txt_path = save_path + '/Analysis Results/diary.txt'

    print("Saving image and writing to diary")
    diary = open(txt_path, "a+")
    diary_text = (now, np.round(best, 2), np.round(int_error, 2))
    diary.write("Time: %s. Fit: %s. Error: %s. \n" % diary_text)
    diary.close()
    plt.savefig(pic_path, dpi = 500)

    stop = time.clock()
    final = time.clock()
    print("Saving results took " + str(round(stop - start, 2)) + " seconds")
    print("Total runtime: " + str(round(final - initial, 2)) + " seconds")
    print("Ready for the next shot!")
    print(" ")
    shot += 1
