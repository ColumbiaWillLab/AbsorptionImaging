"""Shot-related objects (holds raw and calculated image data for a shot)"""
from __future__ import print_function
from __future__ import division

import logging

import imageio
import numpy as np

from boltons.cacheutils import cachedproperty
from scipy.ndimage import filters
from lmfit import Model
from lmfit.models import GaussianModel, ConstantModel

from fitting.utils import ravel, gaussian_2D


class Shot(object):
    """A single shot (3 bmp) sequence"""

    kernel = 3  # for gaussian smoothing
    pixelsize = 3.75e-3  # 3.75 um, reported in mm.

    def __init__(self, bmp_paths):
        logging.info("Reading image data into arrays")
        bmps = map(imageio.imread, bmp_paths)
        bmps = map(lambda x: x.astype("int16"), bmps)  # prevent underflow
        (self.data, self.beam, self.dark) = bmps
        self.shape = self.data.shape

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
        """Create a meshgrid: each pixel is (3.75 um) x (3.75 um); images
        have resolution (964 p) x (1292 p) --> (3.615 mm) x (4.845 mm)"""
        y, x = np.mgrid[: self.height, : self.width]  # mgrid is reverse of meshgrid
        return x, y

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
        logging.info("Performing background subtraction")
        atoms = np.subtract(self.data, self.dark)
        light = np.subtract(self.beam, self.dark)

        # If the light data is below some threshold, we assume that any
        # atom data at this location is invalid and treat as if no transmission.
        # The threshold value was selected experimentally
        threshold = 7
        transmission = np.divide(atoms, light, where=light > threshold)
        transmission[light <= threshold] = 1

        return transmission

    @cachedproperty
    def absorption(self):
        return 1 - self.transmission

    @property
    def transmission_roi(self):
        return self.transmission

    @property
    def absorption_roi(self):
        return self.absorption

    @cachedproperty
    def peak(self):
        """Returns x, y, z of brightest pixel in absorption ROI"""
        y, x = np.unravel_index(np.argmax(self.absorption_roi), self.shape)
        z = self.absorption_roi[y, x]
        logging.info("Finding transmission peak - x: %i, y: %i, z: %i", x, y, z)
        return x, y, z

    @cachedproperty
    def twoD_gaussian(self):
        """Returns an LMFIT ModelResult of the 2D Gaussian for absorption ROI"""
        x, y = self.meshgrid
        x0, y0, A = self.peak

        model = Model(ravel(gaussian_2D), independent_vars=["x", "y"])
        model.set_param_hint("A", value=A, min=0, max=2)
        model.set_param_hint("x0", value=x0, min=0, max=self.width)
        model.set_param_hint("y0", value=y0, min=0, max=self.height)
        model.set_param_hint("sx", min=1, max=self.width)
        model.set_param_hint("sy", min=1, max=self.height)
        model.set_param_hint("theta", min=-np.pi / 2, max=np.pi / 2)
        model.set_param_hint("z0", min=-1, max=1)

        result = model.fit(
            np.ravel(self.absorption_roi[::5]),
            x=x[::5],
            y=y[::5],
            sx=100,
            sy=100,
            theta=0,
            z0=0,
            fit_kws={"maxfev": 200},
        )
        logging.info(result.fit_report())
        return result

    @cachedproperty
    def best_fit_lines(self):
        bp_2D = self.twoD_gaussian.best_values
        h_data = self.absorption_roi[np.rint(bp_2D["y0"]).astype("int32"), :]
        v_data = self.absorption_roi[:, np.rint(bp_2D["x0"]).astype("int32")]
        return h_data, v_data

    @cachedproperty
    def oneD_gaussians(self):
        bp_2D = self.twoD_gaussian.best_values
        h_data, v_data = self.best_fit_lines

        model = GaussianModel() + ConstantModel()
        model.set_param_hint("amplitude", value=bp_2D["A"])
        model.set_param_hint("center", min=0, max=np.max(self.shape))
        model.set_param_hint("sigma", min=0, max=np.max(self.shape))
        model.set_param_hint("c", value=0, min=-0.1, max=0.3)

        h_result = model.fit(
            h_data, x=np.arange(h_data.shape[0]), center=bp_2D["x0"], sigma=bp_2D["sx"]
        )
        v_result = model.fit(
            v_data, x=np.arange(v_data.shape[0]), center=bp_2D["y0"], sigma=bp_2D["sy"]
        )

        return h_result, v_result
