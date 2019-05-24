"""Shot-related objects (holds raw and calculated image data for a shot)"""
from __future__ import print_function
from __future__ import division

import os
import time
import shutil
import imageio
import numpy as np

from boltons.cacheutils import cachedproperty
from scipy.ndimage import filters


def _move_raw_images(bmp_paths):
    """Move (and delete) processed bmp to "Raw Data" folder"""
    garbage_path = "../Raw Data/"
    now = time.strftime("%Y%m%d-%H%M%S")
    pic_num = 1

    for path in bmp_paths:
        name = "Raw_%s_%s.bmp" % (now, str(pic_num))
        os.rename(path, name)
        shutil.copy2(name, garbage_path)
        pic_num += 1
        os.remove(name)


class Shot:
    """A single shot (3 bmp) sequence"""

    kernel = 3  # for gaussian smoothing
    pixelsize = 3.75e-3  # 3.75 um, reported in mm.

    @property
    def height(self):
        """Pixel height of each BMP"""
        return self.shape[0]

    @property
    def width(self):
        """Pixel width of each BMP"""
        return self.shape[1]

    @property
    def pixels(self):
        return [0, Shot.pixelsize * self.width, Shot.pixelsize * self.height, 0]

    @cachedproperty
    def meshgrid(self):
        # create a meshgrid: each pixel is (3.75 um) x (3.75 um); images
        # have resolution (964 p) x (1292 p) --> (3.615 mm) x (4.845 mm)
        width, height = self.width, self.height
        x = np.linspace(0, width, width)
        y = np.linspace(0, height, height)
        return np.meshgrid(x, y)

    @cachedproperty
    def transmission(self):
        """
        INPUT: 3 image arrays (atoms, beam, dark-field)
        PROCESSING:
        - subtract background from both data and beam arrays
        - divide absorption data by beam background to get the transmission t^2
         - apply a Gaussian kernel filter to smooth out remaining noise
        OUTPUT: numpy array containing transmission (0 < t^2 < 1) values
        """
        print("Performing background subtraction")
        atoms = np.subtract(self.data, self.dark)
        light = np.subtract(self.beam, self.dark)

        # If the light data is below some threshold, we assume that any
        # atom data at this location is invalid and treat as if no transmission.
        # The threshold value was selected experimentally
        threshold = 7
        transmission = np.divide(atoms, light, where=light > threshold)
        transmission[light <= threshold] = 1

        print("Applying Gaussian filter of size ", Shot.kernel)
        transmission = filters.gaussian_filter(transmission, Shot.kernel)
        return transmission

    def __init__(self, bmp_paths):
        print("Reading image data into arrays")
        bmps = map(imageio.imread, bmp_paths)
        (self.data, self.beam, self.dark) = bmps
        self.shape = self.data.shape

        _move_raw_images(bmp_paths)
