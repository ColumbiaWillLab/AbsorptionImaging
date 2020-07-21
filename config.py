import configparser
import numpy as np


class Config(configparser.ConfigParser):
    """Holds configuration settings in a file using configparser"""

    units = {
        "camera": {"pixel_size": "mm"},
        "beam": {
            "wavelength": "nm",
            "magnification": "x",
            "detuning": "MHz",
            "linewidth": "MHz",
        },
    }

    def __init__(self, filename):
        self.filename = filename
        super().__init__()
        self.read(self.filename)

        # Non-persistent attributes
        self.fit = True
        self.fit_2D = True
        self.roi_enabled = False
        self.three_roi_enabled = False
        self.fix_center = False
        self.fluor = False

    def save(self):
        """Save current state to file."""
        with open(self.filename, "w") as configfile:
            self.write(configfile, space_around_delimiters=False)

    ##### Camera Settings #####
    @property
    def pixel_size(self):
        """Camera pixel size in mm."""
        return self.getfloat("camera", "pixel_size") * 1e-3

    ##### Beam Settings #####
    @property
    def magnification(self):
        """Imaging beam magnification ratio through optical path to camera."""
        return self.getfloat("beam", "magnification")

    @property
    def physical_scale(self):
        """Pixel to real-space size in mm."""
        return self.pixel_size * (1/self.magnification)

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

    ##### Plot Settings #####
    @property
    def colormap(self):
        """Numpy colormap name."""
        return self.get("plot", "colormap")

    ##### Atom Settings #####
    @property
    def repump_time(self):
        """Repump time in ms."""
        return self.getfloat("atoms", "repump_time")

    @property
    def atom_mass(self):
        """Atom mass in kg."""
        return self.getfloat("atoms", "mass")

    ##### Fit Settings #####
    @property
    def fit_optical_density(self):
        """If True, fits are done against the atom density (-ln(OD))."""
        try:
            return self.getboolean("fit", "fit_optical_density")
        except configparser.NoOptionError:
            return False

    @fit_optical_density.setter
    def fit_optical_density(self, val):
        self["fit"]["fit_optical_density"] = str(val)

    @property
    def fix_theta(self):
        """If True, 2D Gaussian fit restricts theta to 0 (hor/ver)."""
        try:
            return self.getboolean("fit", "fix_theta")
        except configparser.NoOptionError:
            return False

    @fix_theta.setter
    def fix_theta(self, val):
        self["fit"]["fix_theta"] = str(val)

    @property
    def fix_z0(self):
        """If True, 2D Gaussian fit restricts z0 to 0."""
        try:
            return self.getboolean("fit", "fix_z0")
        except configparser.NoOptionError:
            return False

    @fix_z0.setter
    def fix_z0(self, val):
        self["fit"]["fix_z0"] = str(val)

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
    def threeroi(self):
        """Tuple of (x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5) defining three regions of interests (0,1), (2,3), (4,5) for atom counting and subsequent subtraction."""
        try:
            return tuple(map(int, self.get("fit","threeroi").split(",")))
        except (ValueError, configparser.NoOptionError):
            return None

    @threeroi.setter
    def threeroi(self,tup):
        self["fit"]["threeroi"] = ",".join(map(str, tup))

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

    @property
    def center(self):
        """Tuple of (x, y) - 2D Gaussian fit restricted to this center point."""
        try:
            return tuple(map(float, self.get("fit", "center").split(",")))
        except (ValueError, configparser.NoOptionError):
            return None

    @center.setter
    def center(self, tup):
        self["fit"]["center"] = ",".join(map(str, tup))

    @property
    def logdict(self):
        """Returns dictionary of all relevant config parameters"""
        return {"fittedshot" : self.fit,
                "fittedshot2D" : self.fit_2D,
                "roi_enabled" : self.roi_enabled,
                "three_roi_enabled" : self.three_roi_enabled,
                "magnification(x)" : self.magnification,
                "pixel_size(mm)" : self.pixel_size,
                "wavelength(nm)" : self.wavelength,
                "detuning(MHz)" : self.detuning,
                "linewidth(MHz)" : self.linewidth,
                "repump_time(ms)" : self.repump_time,
                "atom_mass(kg)" : self.atom_mass
            }

    def program_name(self):
        try:
            return self.get("program", "name")
        except configparser.NoOptionError:
            return "Absorption Imaging"

config = Config("config.ini")
