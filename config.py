import configparser
import numpy as np


class Config(configparser.ConfigParser):
    """Holds configuration settings in a file using configparser"""

    def __init__(self, filename):
        self.filename = filename
        super().__init__()
        self.read(self.filename)

        # Non-persistent attributes
        self.fit = True
        self.roi_enabled = False

    def save(self):
        """Save current state to file."""
        with open(self.filename, "w") as configfile:
            self.write(configfile, space_around_delimiters=False)

    @property
    def pixel_size(self):
        """Camera pixel size in mm."""
        return self.getfloat("camera", "pixel_size") * 1e-3

    @property
    def magnification(self):
        """Imaging beam magnification ratio through optical path to camera."""
        return self.getfloat("beam", "magnification")

    @property
    def physical_scale(self):
        """Pixel to real-space size in mm."""
        return self.pixel_size * self.magnification

    @property
    def wavelength(self):
        """Imaging beam wavelength in nm."""
        return self.getfloat("beam", "wavelength") * 1e-9

    @property
    def detuning(self):
        """Imaging beam detuning in angular MHz."""
        return self.getfloat("beam", "detuning") * 2 * np.pi

    @property
    def linewidth(self):
        """Imaging beam linewidth in angular MHz."""
        return self.getfloat("beam", "linewidth") * 2 * np.pi

    @property
    def colormap(self):
        """Numpy colormap name."""
        return self.get("plot", "colormap")

    @property
    def repump_time(self):
        """Repump time in ms."""
        return self.getfloat("atoms", "repump_time")

    @property
    def atom_mass(self):
        """Atom mass in kg."""
        return self.getfloat("atoms", "mass")

    @property
    def fix_theta(self):
        """If True, 2D Gaussian fit restricts theta to 0 (hor/ver)."""
        return self.getboolean("fit", "fix_theta")

    @property
    def roi(self):
        """Tuple of (x0, y0, x1, y1) defining the region of interest for fitting."""
        try:
            return tuple(map(int, self.get("fit", "roi").split(",")))
        except (ValueError, configparser.NoOptionError):
            return None

    @roi.setter
    def roi(self, tup):
        self["fit"]["roi"] = ",".join(map(str, tup))

    @property
    def tof(self):
        """List of time of flight shot times in ms for temperature fitting."""
        try:
            return list(map(float, self.get("fit", "tof").split(",")))
        except (ValueError, configparser.NoOptionError):
            return None

    @tof.setter
    def tof(self, arr):
        self["fit"]["tof"] = ",".join(map(str, arr))


config = Config("config.ini")
