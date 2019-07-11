"""Shot-related objects (holds raw and calculated image data for a shot)"""
import logging

import imageio
import numpy as np
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

from boltons.cacheutils import cachedproperty
from scipy.stats.distributions import chi2
from scipy.ndimage import gaussian_filter
from lmfit import Model
from lmfit.models import GaussianModel, ConstantModel

from fitting.utils import ravel, gaussian_2D
from utils.geometry import clipped_endpoints

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

    def meshgrid(self, roi=False):
        """Returns a meshgrid with the whole image dimensions. The meshgrid is an (x, y) tuple of
        numpy matrices whose pairs reference every coordinate in the image."""
        if roi and config.roi_enabled and config.roi:
            x0, y0, x1, y1 = config.roi
            y, x = np.mgrid[y0:y1, x0:x1]
        else:
            y, x = np.mgrid[: self.height, : self.width]  # mgrid is reverse of meshgrid
        return x, y

    @cachedproperty
    def transmission_raw(self):
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

        return transmission

    @cachedproperty
    def absorption_raw(self):
        """Raw absorption data"""
        return 1 - self.transmission_raw

    @cachedproperty
    def transmission(self):
        """The transmission image for fitting (and other derived data) - filtered/clipped"""
        return gaussian_filter(
            np.clip(self.transmission_raw, a_min=0, a_max=1), sigma=3
        )

    @cachedproperty
    def absorption(self):
        """The "inverse" of the transmission (assuming max transmission is 1)"""
        return 1 - self.transmission

    @cachedproperty
    def transmission_roi(self):
        """Transmission pixel matrix bounded by the region of interest."""
        if config.roi_enabled and config.roi:
            x0, y0, x1, y1 = config.roi
            return self.transmission[y0:y1, x0:x1]
        else:
            return self.transmission

    @cachedproperty
    def absorption_roi(self):
        """Absorption pixel matrix bounded by the region of interest."""
        return 1 - self.transmission_roi

    @cachedproperty
    def peak(self):
        """Returns x, y, z of brightest pixel in absorption ROI"""
        y, x = np.unravel_index(
            np.argmax(self.absorption_roi), self.absorption_roi.shape
        )
        z = self.absorption_roi[y, x]
        if config.roi_enabled and config.roi:
            x += config.roi[0]
            y += config.roi[1]
        logging.info("Finding transmission peak - x: %i, y: %i, z: %i", x, y, z)
        return x, y, z

    @cachedproperty
    def twoD_gaussian(self):
        """Returns an LMFIT ModelResult of the 2D Gaussian for absorption ROI"""
        logging.info("Running 2D fit")
        x, y = self.meshgrid(roi=False)
        x0, y0, A = self.peak

        model = Model(ravel(gaussian_2D), independent_vars=["x", "y"])
        model.set_param_hint("A", value=A, min=0, max=2)

        if config.roi_enabled and config.roi:
            x0, y0, x1, y1 = config.roi
            model.set_param_hint("x0", value=x0, min=x0, max=x1)
            model.set_param_hint("y0", value=y0, min=y0, max=y1)
        else:
            model.set_param_hint("x0", value=x0, min=0, max=self.width)
            model.set_param_hint("y0", value=y0, min=0, max=self.height)

        model.set_param_hint("sx", min=1, max=self.width)
        model.set_param_hint("sy", min=1, max=self.height)
        model.set_param_hint(
            "theta", min=-np.pi / 4, max=np.pi / 4, vary=not config.fix_theta
        )
        model.set_param_hint("z0", min=-1, max=1)

        result = model.fit(
            np.ravel(self.absorption[::5]),
            x=x[::5],
            y=y[::5],
            sx=100,
            sy=100,
            theta=0,
            z0=0,
            # scale_covar=False,
            fit_kws={"maxfev": 100, "xtol": 1e-7},
        )
        logging.info(result.fit_report())
        return result

    @cachedproperty
    def contour_levels(self):
        """Returns the 0.5-sigma, 1-sigma, and 1.5-sigma z values of the 2D Gaussian model."""
        bp_2D = self.twoD_gaussian.best_values
        x0, y0, sx, sy = (bp_2D[k] for k in ("x0", "y0", "sx", "sy"))
        sx_pts = x0 + sx * np.array([1.5, 1.0, 0.5])
        sy_pts = y0 + sy * np.array([1.5, 1.0, 0.5])
        x, y = np.meshgrid(sx_pts, sy_pts)

        contour_levels = self.twoD_gaussian.eval(x=x, y=y).reshape((3, 3))
        return np.diag(contour_levels)

    @cachedproperty
    def sigma_mask(self):
        """Returns a numpy mask of pixels within the 2-sigma limit of the model (no ROI)"""
        bp_2D = self.twoD_gaussian.best_values
        x0, y0, a, b, theta = (bp_2D[k] for k in ("x0", "y0", "sx", "sy", "theta"))
        y, x = np.ogrid[0 : self.height, 0 : self.width]

        # https://math.stackexchange.com/a/434482
        maj_axis = np.square((x - x0) * np.cos(theta) - (y - y0) * np.sin(theta))
        min_axis = np.square((x - x0) * np.sin(theta) + (y - y0) * np.cos(theta))
        bound = chi2.ppf(0.886, df=2)

        array = np.zeros(self.shape, dtype="bool")
        array[maj_axis / np.square(a) + min_axis / np.square(b) <= bound] = True
        return array

    @cachedproperty
    def slice_coordinates(self):
        params = self.twoD_gaussian.best_values
        x_c = params["x0"]
        y_c = params["y0"]
        m = np.tan(params["theta"])

        x0_h, y0_h = clipped_endpoints(0, x_c, y_c, m, self.shape[0])
        x1_h, y1_h = clipped_endpoints(self.shape[1], x_c, y_c, m, self.shape[0])

        y0_v, x0_v = clipped_endpoints(0, y_c, x_c, -m, self.shape[1])
        y1_v, x1_v = clipped_endpoints(self.shape[0], y_c, x_c, -m, self.shape[1])

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
        h_data = self.absorption_raw[ys_h.astype("int"), xs_h.astype("int")]
        v_data = self.absorption_raw[ys_v.astype("int"), xs_v.astype("int")]
        return (xs_h, ys_h, h_data), (xs_v, ys_v, v_data)

    @cachedproperty
    def atom_number(self):
        """Calculates the total atom number from the transmission ROI values."""
        # sodium and camera parameters
        sigma_0 = (3 / (2 * np.pi)) * np.square(config.wavelength)  # cross-section
        sigma = sigma_0 * np.reciprocal(
            1 + np.square(config.detuning / (config.linewidth / 2))
        )  # off resonance
        area = np.square(config.physical_scale * 1e-3)  # pixel area in SI units

        if "twoD_gaussian" in self.__dict__:
            data = self.transmission[self.sigma_mask]
        else:
            data = self.transmission

        density = -np.log(data, where=data > 0)
        return (area / sigma) * np.sum(density) / 0.866  # Divide by 1.5-sigma area

    def warm_cache(self, fit=True):
        self.transmission_roi
        if fit:
            self.twoD_gaussian
            self.atom_number
        return True

    def plot(self, fig, **kw):
        fig.clf()
        norm = (-0.1, 1.0)
        color_norm = colors.Normalize(*norm)

        ratio = [1, 9]
        gs = gridspec.GridSpec(2, 2, width_ratios=ratio, height_ratios=ratio)

        image = fig.add_subplot(gs[1, 1])
        image.imshow(self.absorption_raw, cmap=config.colormap, norm=color_norm)

        if config.roi_enabled and config.roi:
            x0, y0, x1, y1 = config.roi
            roi = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor="r", facecolor="none"
            )
            image.add_patch(roi)

        if kw.get("fit", True):
            x, y = self.meshgrid(roi=False)
            (xs_h, ys_h, z_h), (xs_v, ys_v, z_v) = self.best_fit_lines

            image.plot(*self.slice_coordinates[0], "-", linewidth=0.3)
            image.plot(*self.slice_coordinates[1], "-", linewidth=0.3)

            # These plots are already clipped, but this gets rid of the extra padding
            image.set_xlim([0, self.shape[1]])
            image.set_ylim([self.shape[0], 0])

            h_x = np.arange(z_h.shape[0])
            h_y = self.twoD_gaussian.eval(x=xs_h, y=ys_h)
            v_y = np.arange(z_v.shape[0])
            v_x = self.twoD_gaussian.eval(x=xs_v, y=ys_v)

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
                    self.twoD_gaussian.eval(x=x, y=y).reshape(self.shape),
                    levels=self.contour_levels,
                    cmap="magma",
                    linewidths=1,
                    norm=color_norm,
                )
            except ValueError:
                logging.error(
                    "Plotting contour levels failed. Likely because no clear gaussian."
                )

        return fig
