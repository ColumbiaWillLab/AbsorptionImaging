##################################################
######## 0: CLASSES, FUNCTIONS, LIBRARIES ########
##################################################

import logging
from watchdog.observers import Observer

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

    # build the zoomed-in array
    zoomed = []
    for row in data[ymin:ymax]:
        new_row = row[xmin:xmax]
        zoomed.append(new_row)

    x_ext = np.round(dim[1] - dim[0], 2)
    y_ext = np.round(dim[2] - dim[3], 2)
    print("Zooming in: (" +str(x_ext)+" p) x ("+str(y_ext)+" p)")
    return zoomed, dim

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
# easy lines
def easy_lines(x, x0, y0):
    """
    INPUT: lists of domain values and the center (x0, y0)
    OUTPUT: two lists of coordinates (x_hor, y_hor) and (x_ver, y_ver)
    lying along the horizontal & vertical axes of the Gaussian.
    """

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
# easy Gaussian :)
def gaus(x,a,x0,sigma,offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

def read_pgm(pgmfile):
    """
    INPUT: .pgm file containing grayscale pixel depths from 1 to 255.
    OUTPUT: numpy array containing the pixel values, and its dimensions.
    """

    # Read header contents
    magic_num = pgmfile.readline()
    dimensions = pgmfile.readline()
    width = int(dimensions.split()[0])
    height = int(dimensions.split()[1])
    max_val = pgmfile.readline()

    # Put values in their places
    raw_data = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            value = ord(pgmfile.read(1))
            raw_data[y][x] = int(value)
    return width, height, raw_data

EVENT_TYPE_MOVED = 'moved'
EVENT_TYPE_DELETED = 'deleted'
EVENT_TYPE_CREATED = 'created'
EVENT_TYPE_MODIFIED = 'modified'

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
        # what = 'directory' if event.is_directory else 'file'
        filename = event.src_path

        despacito.append(filename)
        count += 1 # tick the counter for each new file
        print("Created " + filename + "; count = " + str(count))
        return despacito

        # execute arbitrary code!
    def on_deleted(self, event):
        global count
        super(LoggingEventHandler, self).on_deleted(event)

        # what = 'directory' if event.is_directory else 'file'
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

##################################################
######### 1: READ FILES AND CREATE DATA ##########
##################################################

# main watchdog loop - counts to 3
"""
EVENT_TYPE_MOVED = 'moved'
EVENT_TYPE_DELETED = 'deleted'
EVENT_TYPE_CREATED = 'created'
EVENT_TYPE_MODIFIED = 'modified'
hp.despacito = []

if __name__ == "__main__":
    print("Watching for new files")
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    event_handler = hp.LoggingEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
            if hp.count >= 3:
                print("3 new images: watchdog naptime")
                raise hp.MyExcept()
    except hp.MyExcept:
        observer.stop()
    observer.join()
"""

# retrieve names of newly created image files
"""
despacito2 = []
for meme in hp.despacito:
    meme = meme[2:]
    despacito2.append(meme)
"""

# open files for each shot: atoms + laser --> laser --> background
"""
print("Writing image data into arrays:")
data = open(despacito2[0], 'rb')
beam = open(despacito2[1], 'rb')
dark = open(despacito2[2], 'rb')

# write data from each image into a large array
print("- data")
width, height, data = read_pgm(data)
print("- beam")
width, height, beam = read_pgm(beam)
print("- dark")
width, height, dark = read_pgm(dark)
"""

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

##################################################
######### 2: GAUSSIAN FITTING (2D IMAGE) #########
##################################################

"""
# allow the user to select a region of interest
plt.figure(1)
plt.imshow(transmission, cmap = 'inferno', extent = pixels)
plt.colorbar()
ROI = hp.roipoly(roicolor = 'b')
mask, coords = ROI.getMask(transmission)

# zoom into the region of interest; create a new meshgrid for plotting
zoomed, dim_z = zoom_in(transmission, params, 5)
x_z = np.linspace(dim_z[0], dim_z[1], len(zoomed[0]))
y_z = np.linspace(dim_z[3], dim_z[2], len(zoomed))
(x_z, y_z) = np.meshgrid(x_z, y_z)

# run the zoomed-in fit and compute relative error
fine_fit, best = iterfit(residual, params, x_z, y_z, zoomed, 1)
fine_error = (zoomed - fine_fit) / zoomed
area = (dim_z[1] - dim_z[0]) * (dim_z[2] - dim_z[3])
int_f_error = (np.sum((fine_error)**2) / area) * 100
print("Integrated error: " + str(round(int_f_error, 2)))

# generate final-fit transmission data; compute relative error
params0 = list2params(best)
fit_data = 1 - gaussian(params0, x,y)
final_error = (transmission - fit_data) / transmission
"""

# (x_min, x_max, y_min, y_max) = coords

##################################################
########### pi: POOR MAN'S GAUSSIAN FIT ##########
##################################################

"""
print("pi: POOR MAN'S GAUSSIAN FITTING")
print("-------------------------------")
start = time.clock()

(fx, fy) = (10, 10) # reduce the image resolution
print("Mode: " + mode)

# coarsen the array and guess the fit parameters
coarse = de_enhance(transmission, fx, fy)

# create a new (coarse) meshgrid for plotting
x_c = np.linspace(fx, len(coarse[0])*fx, len(coarse[0]))
y_c = np.linspace(fx, len(coarse) * fy, len(coarse))
(x_c, y_c) = np.meshgrid(x_c, y_c)

# take an "intelligent" guess and run the preliminary fit
(y0, x0, peak) = peak_find(coarse) # guess an initial center point
(amp, z0) = (transmission[0][0] - peak, 1 - transmission[0][0])
guess = [amp, x0, y0, 200, 200, 0.5*math.pi/4, z0]

# compute mean matrices in x and in y
meansX = np.mean(transmission, axis=0)
meansY = np.mean(transmission, axis=1)
(X, Y) = (np.argmin(meansX), np.argmin(meansY))
print(len(meansX))
print(len(meansY))
print(X)
print(Y)

# initialize fit arrays
Yfit = transmission[:,X]
Xfit = transmission[Y,:]
Xrange = range(0, Xfit.shape[0])
Yrange = range(0, Yfit.shape[0])

# do the fitting
# parameters: [Amp, mean, stdev, offset]
popt_x,pcov_x = curve_fit(gaus, Xrange, Xfit, p0=[-1, X, -100, Xfit.max()])
popt_y,pcov_y = curve_fit(gaus, Yrange, Yfit, p0=[-1, Y, -100, Yfit.max()])

print(popt_x)
print(popt_y)

stop = time.clock()
print("Simple fitting took " + str(round(stop - start, 2)) + " seconds")
print(" ")
"""

##################################################
######### 5: GRAPHS, PLOTS, AND PICTURES #########
##################################################

"""
# axis scales for different graphs
pixels = [0, width, height, 0]
sampled = [0, len(coarse[0])*fx, len(coarse)*fy, 0]
closeup = dim_z
"""

##################################################
######### 7: SAVE AND DISPLAY FINAL PLOT #########
##################################################

# create figure: 3x3 grid (boxes labeled 0-8) containing:
# - title (1);
# - best-fit parameter outputs (8)
# - horizontal (4) and vertical (6) plots
# - transmission plot, in color (7)
# - empty boxes: 0, 2, 3, 5

"""
print("Painting Gaussian fits in oil")
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

# poor man's fit parameters: convert to text
(Ax, x_0, w_x, z0_x) = (str(popt_x[0]), str(popt_x[1]), str(2*popt_x[2]), str(popt_x[3]))
(Ay, y_0, w_y, z0_y) = (str(popt_y[0]), str(popt_y[1]), str(2*popt_y[2]), str(popt_y[3]))

text1 = 'Best-fit parameters:'
text2 = 'A_x = ' + Ax
text3 = 'A_y = ' + Ay
text4 = 'x_0 = ' + x_0
text5 = 'y_0 = ' + y_0
text6 = 'w_x = '+ w_x
text7 = 'w_y = '+ w_y
# text8 = 'theta = '+ theta + ' rad'
text9 = 'N = ' + str(np.round(atom_num/1000000.0, 2)) + ' million'

# best-fit parameters: display
ax8 = plt.subplot(gs[8])
plt.axis('off')
plt.text(0, 0.9, text1)
plt.text(0, 0.8, text2)
plt.text(0, 0.7, text3)
plt.text(0, 0.6, text4)
plt.text(0, 0.5, text5)
plt.text(0, 0.4, text6)
plt.text(0, 0.3, text7)
# plt.text(0, 0.2, text8)
plt.text(0, 0.2, text9)

# horizontal and vertical 1D fits
ax4 = plt.subplot(gs[4])

plt.plot(Xrange, gaus(Xrange,*popt_x), 'ko', markersize = 1)
plt.plot(Xrange, Xfit, 'r', linewidth = 0.5)
pixels = [0, width, height, 0]
plt.xlim(pixels[0], pixels[1])
plt.ylim(0, 1)

ax6 = plt.subplot(gs[6])
plt.plot(gaus(Yrange,*popt_y),Yrange, 'ko', markersize = 1)
plt.plot(Yfit, Yrange, 'r', linewidth = 0.5)
plt.xlim(0, 1)
plt.ylim(pixels[0], pixels[1])


plt.ylim(pixels[2], pixels[3])
plt.gca().axes.get_yaxis().set_visible(False)

# transmission plot with axis lines and zoom box
ax7 = plt.subplot(gs[7])
norm = plt.Normalize(0, 1)
plt.imshow(1 - transmission, cmap = 'inferno', norm = norm, extent = pixels)

x_val = np.linspace(0, width, width)
y_val = np.linspace(0, height, height)
(x_hor, y_hor, x_ver, y_ver) = easy_lines(x_val, popt_x[1], popt_y[1])
plt.plot(x_hor, y_hor, color = 'r', linewidth = 0.5)
plt.plot(x_ver, y_ver, color = 'r', linewidth = 0.5)

plt.xlim(pixels[0], pixels[1])
plt.ylim(pixels[2], pixels[3])

# save best-fit parameters and image to files
save_path = './'
pic_path = save_path + '/Analysis Results/fig'+str(shot)+'.png'
txt_path = save_path + '/Analysis Results/diary.txt'

print("Saving image")
plt.savefig(pic_path, dpi = 500)
plt.clf()
plt.close()

stop = time.clock()
final = time.clock()
print("Saving results took " + str(round(stop - start, 2)) + " seconds")
print("Total runtime: " + str(round(final - initial, 2)) + " seconds")
print("Ready for the next shot!")
shot += 1
"""
