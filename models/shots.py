"""Shot-related objects (holds raw and calculated image data for a shot)"""
import logging

import imageio
import numpy as np
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

from boltons.cacheutils import cachedproperty
from scipy.ndimage import median_filter
from scipy.stats.distributions import chi2
from lmfit import Model
from lmfit.models import GaussianModel, ConstantModel

from fitting.utils import ravel, gaussian_2D

from config import config


class Shot:
    """A single shot (3 bmp) sequence"""

    def __init__(self, name, bmp_paths):
        logging.info("Reading image data into arrays")
        bmps = map(imageio.imread, bmp_paths)
        bmps = map(lambda x: x.astype("int16"), bmps)  # prevent underflow
        (self.data, self.beam, self.dark) = bmps
        self.shape = self.data.shape
        self.name = name

    @property
    def height(self):
        """Pixel height of each BMP"""
        return self.shape[0]

    @property
    def width(self):
        """Pixel width of each BMP"""
        return self.shape[1]

    @cachedproperty
    def meshgrid(self):
        """Returns a meshgrid with the whole image dimensions. The meshgrid is an (x, y) tuple of
        numpy matrices whose pairs reference every coordinate in the image."""
        y, x = np.mgrid[: self.height, : self.width]  # mgrid is reverse of meshgrid
        return x, y

    @cachedproperty
    def transmission(self):
        """Returns the beam and dark-field compensated transmission image. Dark-field is subtracted
        from both the atom image and the beam image, and the atom image is divided by the beam
        image, giving the transmission t^2. The values should optimally lie in the range of [0, 1]
        but can realistically be in the range of [-0.1, 1.5] due to noise and beam variation across
        images."""
        logging.info("Performing background subtraction")
        atoms = np.subtract(self.data, self.dark)
        light = np.subtract(self.beam, self.dark)

        # If the light data is below some threshold, we assume that any
        # atom data at this location is invalid and treat as if no transmission.
        # The threshold value was selected experimentally
        threshold = 7
        transmission = np.divide(atoms, light, where=light > threshold)
        transmission[light <= threshold] = 1

        # transmission = median_filter(transmission, size=20)

        return transmission

    @cachedproperty
    def absorption(self):
        """The "inverse" of the transmission (assuming max transmission is 1)"""
        return 1 - self.transmission

    @cachedproperty
    def transmission_roi(self):
        """Transmission pixel matrix bounded by the region of interest."""
        return self.transmission

    @cachedproperty
    def absorption_roi(self):
        """Absorption pixel matrix bounded by the region of interest."""
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
        logging.info("Running 2D fit")
        x, y = self.meshgrid
        x0, y0, A = self.peak

        model = Model(ravel(gaussian_2D), independent_vars=["x", "y"])
        model.set_param_hint("A", value=A, min=0, max=2)
        model.set_param_hint("x0", value=x0, min=0, max=self.width)
        model.set_param_hint("y0", value=y0, min=0, max=self.height)
        model.set_param_hint("sx", min=1, max=self.width)
        model.set_param_hint("sy", min=1, max=self.height)
        model.set_param_hint("theta", min=-np.pi / 4, max=np.pi / 4)
        model.set_param_hint("z0", min=-1, max=1)

        result = model.fit(
            np.ravel(self.absorption_roi[::5]),
            x=x[::5],
            y=y[::5],
            sx=100,
            sy=100,
            theta=0,
            z0=0,
            # scale_covar=False,
            fit_kws={"maxfev": 100, "xtol": 1.0e-3},
        )
        logging.info(result.fit_report())
        return result

    @cachedproperty
    def contour_levels(self):
        """Returns the 1-sigma, 2-sigma, and 3-sigma z values of the 2D Gaussian model."""
        bp_2D = self.twoD_gaussian.best_values
        x0, y0, sx, sy = (bp_2D[k] for k in ("x0", "y0", "sx", "sy"))
        sx_pts = x0 + sx * np.arange(3, 0, -1)
        sy_pts = y0 + sy * np.arange(3, 0, -1)
        x, y = np.meshgrid(sx_pts, sy_pts)

        contour_levels = self.twoD_gaussian.eval(x=x, y=y).reshape((3, 3))
        return np.diag(contour_levels)

    @cachedproperty
    def two_sigma_mask(self):
        """Returns a numpy mask of pixels within the 2-sigma limit of the model (no ROI)"""
        # TODO: assumes independence, needs covar matrix
        bp_2D = self.twoD_gaussian.best_values
        x0, y0, sx, sy = (bp_2D[k] for k in ("x0", "y0", "sx", "sy"))
        y, x = np.ogrid[-y0 : self.height - y0, -x0 : self.width - x0]
        mask = np.square(x) / np.square(sx) + np.square(y) / np.square(sy) <= chi2.ppf(
            0.95, df=2
        )
        array = np.zeros(self.shape, dtype="bool")
        array[mask] = True
        return array

    @cachedproperty
    def best_fit_lines(self):
        """Gets the absorption ROI values across the horizontal/vertical lines (no theta) of the 2D
        Gaussian fit."""
        bp_2D = self.twoD_gaussian.best_values
        h_data = self.absorption_roi[np.rint(bp_2D["y0"]).astype("int32"), :]
        v_data = self.absorption_roi[:, np.rint(bp_2D["x0"]).astype("int32")]
        return h_data, v_data

    @cachedproperty
    def oneD_gaussians(self):
        """Returns a tuple of (hor, ver) 1D Gaussian ModelResult across the 2D best fit lines."""
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

    @cachedproperty
    def atom_number(self):
        """Calculates the total atom number from the transmission ROI values."""
        # sodium and camera parameters
        sigma_0 = (3 / (2 * np.pi)) * np.square(config.wavelength)  # cross-section
        sigma = sigma_0 * np.reciprocal(
            1 + np.square(config.detuning / (config.linewidth / 2))
        )  # off resonance
        area = np.square(
            config.pixel_size * 1e-3 * config.magnification
        )  # pixel area in SI units

        density = -np.log(self.transmission_roi, where=self.transmission_roi > 0)
        return (area / sigma) * np.sum(density)

    def plot(self, fig):
        fig.clf()
        norm = (-0.1, 1.0)
        color_norm = colors.Normalize(*norm)

        x, y = self.meshgrid
        params = self.twoD_gaussian.best_values
        h, v = self.best_fit_lines
        hfit, vfit = self.oneD_gaussians

        ratio = [1, 9]
        gs = gridspec.GridSpec(2, 2, width_ratios=ratio, height_ratios=ratio)

        image = fig.add_subplot(gs[1, 1])
        image.imshow(self.absorption, cmap=config.colormap, norm=color_norm)

        image.contour(
            self.twoD_gaussian.eval(x=x, y=y).reshape(self.shape),
            levels=self.contour_levels,
            cmap="magma",
            linewidths=1,
            norm=color_norm,
        )

        image.axhline(params["y0"], linewidth=0.3)
        image.axvline(params["x0"], linewidth=0.3)

        h_x = np.arange(h.shape[0])
        h_y = hfit.eval(x=h_x)
        v_y = np.arange(v.shape[0])
        v_x = vfit.eval(x=v_y)

        hor = fig.add_subplot(gs[0, 1])
        hor.plot(h_x, h, "ko", markersize=0.2)
        hor.plot(h_x, h_y, "r", linewidth=0.5)
        hor.set_ylim(*norm)
        hor.get_xaxis().set_visible(False)

        ver = fig.add_subplot(gs[1, 0])
        ver.plot(v, v_y, "ko", markersize=0.2)
        ver.plot(v_x, v_y, "r", linewidth=0.5)
        ver.set_xlim(*norm)
        ver.invert_xaxis()
        ver.get_yaxis().set_visible(False)

        return fig