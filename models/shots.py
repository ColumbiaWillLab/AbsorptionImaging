"""Shot-related objects (holds raw and calculated image data for a shot)"""
import logging
import functools

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import imageio
import numpy as np
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

from boltons.cacheutils import cachedproperty
from scipy.stats.distributions import chi2
from scipy.ndimage import gaussian_filter
from lmfit import Model
from lmfit.models import GaussianModel

from config import config
from utils.fitting import ravel, gaussian_2D
from utils.geometry import clipped_endpoints


class Shot:
    """A single shot (3 bmp) sequence"""

    def __init__(self, name, bmp_paths):
        logging.info("Reading image data into arrays")
        bmps = map(imageio.imread, bmp_paths)
        bmps = map(lambda x: x.astype("int16"), bmps)  # prevent underflow

        (self.data, self.beam, self.dark) = bmps
        self.shape = self.data.shape
        self.name = name

        self.fit_2D = None
        self.fit_1D = None

        # Warm transmission cache and log shape to debug
        logging.debug("Processed transmission: %s", self.transmission.shape)

    def __eq__(self, other):
        return self.name == other.name and np.array_equal(
            self.transmission, other.transmission
        )

    @property
    def height(self):
        """Pixel height of each BMP"""
        return self.shape[0]

    @property
    def width(self):
        """Pixel width of each BMP"""
        return self.shape[1]

    @property
    def meshgrid(self):
        """Returns a meshgrid with the shot dimensions. The meshgrid is an (x, y) tuple of
        numpy matrices whose pairs reference every coordinate in the image."""
        y, x = np.mgrid[0 : self.height, 0 : self.width]
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
        np.clip(transmission, a_min=0, a_max=1, out=transmission)

        return transmission

    @property
    def absorption(self):
        """Raw absorption data"""
        return 1 - self.transmission

    @cachedproperty
    def optical_density(self):
        smoothed_transmission = gaussian_filter(self.transmission, sigma=1)
        od = -np.log(smoothed_transmission, where=smoothed_transmission > 0)
        return od

    @property
    def fit(self):
        # if self.fit_2D is None:
        #     return self.fit_1D
        return self.fit_2D

    def run_fit(self, cfg):
        self.clear_fit()

        if cfg.fit_2D:
            return self.run_fit_2D(cfg)
        else:
            return self.run_fit_1D_summed(cfg)

    def run_fit_2D(self, cfg):
        self.run_fit_1D_summed(cfg)

        roi = None
        if cfg.roi_enabled and cfg.roi:
            roi = cfg.roi

        center = None
        if cfg.fix_center and cfg.center:
            center = cfg.center

        self.fit_2D = ShotFit2D(
            self,
            fit_density=cfg.fit_optical_density,
            roi=roi,
            center=center,
            fix_theta=cfg.fix_theta,
            fix_z0=cfg.fix_z0,
            guess=self.fit_1D.best_values,
        )
        return self.fit_2D

    def run_fit_1D_summed(self, cfg):
        self.fit_1D = ShotFit1DSummed(self, fit_density=cfg.fit_optical_density)
        return self.fit_1D

    def clear_fit(self):
        self.fit_2D = None
        self.fit_1D = None

    @property
    def atom_number(self):
        """Calculates the total atom number from the transmission ROI values."""
        # sodium and camera parameters
        sigma_0 = (3 / (2 * np.pi)) * np.square(config.wavelength)  # cross-section
        sigma = sigma_0 * np.reciprocal(
            1 + np.square(config.detuning / (config.linewidth / 2))
        )  # off resonance
        area = np.square(config.physical_scale * 1e-3)  # pixel area in SI units

        density = self.optical_density
        if self.fit:
            if self.fit.roi:
                x0, y0, x1, y1 = self.fit.roi
                density = density[x0:x1, y0:y1]
            else:
                density = density[self.fit.sigma_mask]

        return (area / sigma) * np.sum(density) / 0.866  # Divide by 1.5-sigma area

    def plot(self, fig, *args, **kwargs):
        fig.clf()

        if config.fit_optical_density:
            nmax = 6.0
            img = self.optical_density
        else:
            nmax = 1.0
            img = self.absorption

        norm = (-0.1, nmax)
        color_norm = colors.Normalize(*norm)

        ratio = [1, 9]
        gs = gridspec.GridSpec(2, 2, width_ratios=ratio, height_ratios=ratio)

        image = fig.add_subplot(gs[1, 1])
        image.imshow(img, cmap=config.colormap, norm=color_norm)

        if config.roi_enabled and config.roi:
            x0, y0, x1, y1 = config.roi
            roi = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor="r", facecolor="none"
            )
            image.add_patch(roi)

        if self.fit:
            x, y = self.meshgrid
            (xs_h, ys_h, z_h), (xs_v, ys_v, z_v) = self.fit.best_fit_lines

            image.plot(*self.fit.slice_coordinates[0], "-", linewidth=0.3)
            image.plot(*self.fit.slice_coordinates[1], "-", linewidth=0.3)

            # These plots are already clipped, but this gets rid of the extra padding
            image.set_xlim([0, self.width])
            image.set_ylim([self.height, 0])

            h_x = np.arange(z_h.shape[0])
            h_y = self.fit.eval(x=xs_h, y=ys_h)
            v_y = np.arange(z_v.shape[0])
            v_x = self.fit.eval(x=xs_v, y=ys_v)

            hor = fig.add_subplot(gs[0, 1])
            hor.plot(h_x, z_h, "ko", markersize=0.2)
            hor.plot(h_x, h_y, "r", linewidth=0.5)
            hor.set_ylim(*norm)
            hor.get_xaxis().set_visible(False)

            ver = fig.add_subplot(gs[1, 0])
            ver.plot(z_v, v_y, "ko", markersize=0.2)
            ver.plot(v_x, v_y, "r", linewidth=0.5)
            ver.set_xlim(*norm)
            ver.invert_xaxis()
            ver.invert_yaxis()
            ver.get_yaxis().set_visible(False)

            try:
                image.contour(
                    self.fit.eval(x=x, y=y).reshape(self.shape),
                    levels=self.fit.contour_levels,
                    cmap=config.colormap,
                    linewidths=1,
                    norm=color_norm,
                )
            except ValueError as err:
                logging.error(
                    "Plotting contour levels failed: %s. Likely because no clear gaussian",
                    self.fit.contour_levels,
                )
                logging.error(err)

        return fig


class ShotFit(ABC):
    def __init__(
        self,
        shot: Shot,
        fit_density: bool = False,
        roi: Optional[Tuple[int, int, int, int]] = None,
        center: Optional[Tuple[int, int]] = None,
        fix_theta: bool = True,
        fix_z0: bool = False,
        guess: Optional[dict] = None,
    ):
        self.shot = shot
        self.fit_density = fit_density
        self.roi = roi
        self.center = center
        self.vary_theta = not fix_theta
        self.vary_z0 = not fix_z0
        self.guess = guess

        self.result = None

        self.fit()

    @classmethod
    def save_result(cls, func):
        """Decorator for the 'fit' method to save result."""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.result = result
            return result

        return wrapper

    @property
    def meshgrid(self):
        """Returns a meshgrid with the shot ROI dimensions. The meshgrid is an (x, y) tuple of
        numpy matrices whose pairs reference every coordinate in the image."""
        if self.roi:
            x0, y0, x1, y1 = self.roi
            y, x = np.mgrid[y0:y1, x0:x1]
        else:
            y, x = self.shot.meshgrid
        return x, y

    @cachedproperty
    def fit_data(self):
        if self.fit_density:
            return self.shot.optical_density
        else:
            return gaussian_filter(self.shot.absorption, sigma=3)

    @property
    def fit_data_raw(self):
        if self.fit_density:
            return self.shot.optical_density
        else:
            return self.shot.absorption

    @property
    def fit_data_roi(self):
        if self.roi:
            x0, y0, x1, y1 = self.roi
            return self.fit_data[y0:y1, x0:x1]

        return self.fit_data

    @property
    def peak(self):
        """Returns x, y, z of brightest pixel in absorption ROI"""
        y, x = np.unravel_index(np.argmax(self.fit_data_roi), self.fit_data_roi.shape)
        z = self.fit_data_roi[y, x]

        if self.roi:
            x += self.roi[0]
            y += self.roi[1]

        logging.info("Finding transmission peak - x: %i, y: %i, z: %i", x, y, z)
        return x, y, z

    @cachedproperty
    def contour_levels(self):
        """Returns the 0.5-sigma, 1-sigma, and 1.5-sigma z values of the 2D Gaussian model."""
        bp_2D = self.best_values
        x0, y0, sx, sy = (bp_2D[k] for k in ("x0", "y0", "sx", "sy"))
        sx_pts = x0 + sx * np.array([1.5, 1.0, 0.5])
        sy_pts = y0 + sy * np.array([1.5, 1.0, 0.5])
        x, y = np.meshgrid(sx_pts, sy_pts)

        contour_levels = self.eval(x=x, y=y).reshape((3, 3))
        return np.diag(contour_levels)

    @property
    def sigma_mask(self):
        """Returns a numpy mask of pixels within the 2-sigma limit of the model (no ROI)"""
        bp_2D = self.best_values
        x0, y0, a, b, theta = (bp_2D[k] for k in ("x0", "y0", "sx", "sy", "theta"))
        y, x = np.ogrid[0 : self.shot.height, 0 : self.shot.width]

        # https://math.stackexchange.com/a/434482
        maj_axis = np.square((x - x0) * np.cos(theta) - (y - y0) * np.sin(theta))
        min_axis = np.square((x - x0) * np.sin(theta) + (y - y0) * np.cos(theta))
        bound = chi2.ppf(0.886, df=2)

        array = np.zeros(self.shot.shape, dtype="bool")
        array[maj_axis / np.square(a) + min_axis / np.square(b) <= bound] = True
        return array

    @cachedproperty
    def slice_coordinates(self):
        params = self.best_values
        x_c = params["x0"]
        y_c = params["y0"]
        m = np.tan(params["theta"])

        x0_h, y0_h = clipped_endpoints(0, x_c, y_c, m, self.shot.height)
        x1_h, y1_h = clipped_endpoints(self.shot.width, x_c, y_c, m, self.shot.height)

        y0_v, x0_v = clipped_endpoints(0, y_c, x_c, -m, self.shot.width)
        y1_v, x1_v = clipped_endpoints(self.shot.height, y_c, x_c, -m, self.shot.width)

        return ((x0_h, x1_h), (y0_h, y1_h)), ((x0_v, x1_v), (y0_v, y1_v))

    @cachedproperty
    def best_fit_lines(self):
        """Gets the absorption values across the horizontal/vertical lines (no theta) of the 2D
        Gaussian fit."""
        (x_h, y_h), (x_v, y_v) = self.slice_coordinates
        h_length = int(np.hypot(x_h[1] - x_h[0], y_h[1] - y_h[0]))
        v_length = int(np.hypot(x_v[1] - x_v[0], y_v[1] - y_v[0]))
        xs_h, ys_h = (
            np.linspace(min(x_h), max(x_h), h_length, endpoint=False),
            np.linspace(min(y_h), max(y_h), h_length, endpoint=False),
        )
        xs_v, ys_v = (
            np.linspace(min(x_v), max(x_v), v_length, endpoint=False),
            np.linspace(min(y_v), max(y_v), v_length, endpoint=False),
        )
        h_data = self.fit_data_raw[ys_h.astype("int"), xs_h.astype("int")]
        v_data = self.fit_data_raw[ys_v.astype("int"), xs_v.astype("int")]
        return (xs_h, ys_h, h_data), (xs_v, ys_v, v_data)

    @abstractmethod
    def fit(self):
        """Implement this method in your derived class. Run the actual fit."""

    @abstractmethod
    def eval(self, *, x, y):
        """Implement this method in your derived class.
        Evaluates the fit at the given coordinates (proxy for ModelResult)."""

    @abstractmethod
    def best_values(self):
        """Implement this property in your derived class. Returns a dictionary of values."""


class ShotFit2D(ShotFit):
    """2D Gaussian fit"""

    @ShotFit.save_result
    def fit(self):
        """Fits a 2D Gaussian against the absorption."""
        logging.info("Running 2D fit...")

        x_mg, y_mg = self.shot.meshgrid
        model = Model(ravel(gaussian_2D), independent_vars=["x", "y"])

        if self.roi:
            x0, y0, x1, y1 = self.roi
        else:
            x0, y0, x1, y1 = 0, 0, self.shot.width, self.shot.height

        x_c, y_c, x_s, y_s, A, vary_center = None, None, 100, 100, 0.5, True
        if self.center:
            x, y = self.center
            if x < x0 or x > x1 or y < y0 or y > y1:
                logging.warning(
                    "Center fix: %s is outside the ROI: %s. Ignoring center!",
                    self.center,
                    self.roi,
                )
            else:
                x_c, y_c = self.center
                vary_center = False
        elif self.roi:
            x_c, y_c = (x1 + x0) / 2, (y1 + y0) / 2

        if not x_c and not y_c:
            if self.guess:
                x_c, y_c, x_s, y_s, A = (
                    self.guess["x0"],
                    self.guess["y0"],
                    self.guess["sx"],
                    self.guess["sy"],
                    self.guess["A"],
                )
            else:
                x_c, y_c, A = self.peak

        if self.fit_density:
            max_A = 6
        else:
            max_A = 1.5

        model.set_param_hint("A", value=A, min=0, max=max_A)
        model.set_param_hint("x0", value=x_c, min=x0, max=x1, vary=vary_center)
        model.set_param_hint("y0", value=y_c, min=y0, max=y1, vary=vary_center)

        model.set_param_hint("sx", min=1, max=self.shot.width)
        model.set_param_hint("sy", min=1, max=self.shot.height)
        model.set_param_hint(
            "theta", min=-np.pi / 4, max=np.pi / 4, vary=self.vary_theta
        )
        model.set_param_hint("z0", min=-1, max=1, vary=self.vary_z0)

        logging.info(
            "Using guess: x0=%.2f, y0=%.2f, sx=%.2f, sy=%.2f, A=%.2f",
            x_c,
            y_c,
            x_s,
            y_s,
            A,
        )
        result = model.fit(
            np.ravel(self.fit_data[::5]),
            x=x_mg[::5],
            y=y_mg[::5],
            sx=x_s,
            sy=y_s,
            theta=0,
            z0=0,
            # scale_covar=False,
            fit_kws={"maxfev": 100, "xtol": 1e-7},
        )
        logging.info(result.fit_report())
        return result

    def eval(self, *, x, y):
        """Evaluates the fit at the given coordinates (proxy for ModelResult)."""
        return self.result.eval(x=x, y=y)

    @property
    def best_values(self):
        return self.result.best_values


class ShotFit1DSummed(ShotFit):
    """1D Gaussian fit computed against the horizontal and vertical sums of the image."""

    @ShotFit.save_result
    def fit(self):
        logging.info("Running 1D summed fit...")

        x_mg, y_mg = np.arange(self.shot.width), np.arange(self.shot.height)
        x_data, y_data = (
            np.sum(self.fit_data, axis=0),
            np.ravel(np.sum(self.fit_data, axis=1)),
        )
        model = GaussianModel()

        x_result = model.fit(x_data, x=x_mg, center=self.shot.width / 2)
        y_result = model.fit(y_data, x=y_mg, center=self.shot.height / 2)

        return x_result, y_result

    def eval(self, *, x, y):
        """Evaluates the fit at the given coordinates (proxy for ModelResult)."""
        x = self.result[0].eval(x=x)
        y = self.result[1].eval(x=y)
        scale = self.shot.width * self.shot.height
        return np.sqrt(np.multiply(np.square(x), np.square(y))) / scale

    @property
    def best_values(self):
        return {
            "A": 1,  # TODO:  calculate
            "x0": self.result[0].best_values["center"],
            "y0": self.result[1].best_values["center"],
            "sx": self.result[0].best_values["sigma"],
            "sy": self.result[1].best_values["sigma"],
            "theta": 0,
            "z0": 0,
        }
