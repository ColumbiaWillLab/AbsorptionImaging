import logging

from collections import deque
from abc import ABC, abstractmethod

import numpy as np

from boltons.cacheutils import cachedproperty
from scipy.stats import linregress

from config import config


class ShotSequence(ABC):
    pass


class TimeOfFlight(ShotSequence):
    KB = 1.38e-23

    def __init__(self, times):
        self.times = times
        self.shots = deque(maxlen=len(times))

    @cachedproperty
    def sigma_x_pixel(self):
        return [s.fit.result.best_values["sx"] for s in self.shots]

    @cachedproperty
    def sigma_y_pixel(self):
        return [s.fit.result.best_values["sy"] for s in self.shots]

    @cachedproperty
    def sigma_x_sq(self):
        return np.square([s * config.physical_scale for s in self.sigma_x_pixel])

    @cachedproperty
    def sigma_y_sq(self):
        return np.square([s * config.physical_scale for s in self.sigma_y_pixel])

    @cachedproperty
    def times_sq(self):
        return np.square(np.array(self.times) + config.repump_time)

    @cachedproperty
    def x_fit(self):
        return linregress(self.times_sq, self.sigma_x_sq)

    @cachedproperty
    def y_fit(self):
        return linregress(self.times_sq, self.sigma_y_sq)

    @cachedproperty
    def x_temp(self):
        return self.x_fit[0] * config.atom_mass / type(self).KB * 1e6

    @cachedproperty
    def y_temp(self):
        return self.y_fit[0] * config.atom_mass / type(self).KB * 1e6

    @cachedproperty
    def x_temp_err(self):
        return self.x_fit[4] * config.atom_mass / type(self).KB * 1e6

    @cachedproperty
    def y_temp_err(self):
        return self.y_fit[4] * config.atom_mass / type(self).KB * 1e6

    @cachedproperty
    def avg_temp(self):
        return np.mean((self.x_temp, self.y_temp))

    @cachedproperty
    def avg_temp_err(self):
        return 0.5 * np.sqrt(np.sum(np.square((self.x_temp_err, self.y_temp_err))))

    @cachedproperty
    def atom_number(self):
        return [s.atom_number for s in self.shots]

    def add(self, shot, cb):
        if len(self.shots) < self.shots.maxlen:
            logging.info(
                "Added shot %s as t=%.2fms to ToF",
                shot.name,
                self.times[len(self.shots)],
            )
            self.shots.append(shot)
            if len(self.shots) == self.shots.maxlen:
                # for shot in self.shots:
                #     shot.warm_cache(fit=True)
                cb(self)

    def plot(self, fig, **kw):
        fig.clf()
        plt = fig.add_subplot()

        color = next(plt._get_lines.prop_cycler)["color"]
        plt.plot(self.times_sq, self.sigma_x_sq, "o", color=color, label="X")
        plt.plot(
            self.times_sq, self.x_fit[0] * self.times_sq + self.x_fit[1], color=color
        )

        color = next(plt._get_lines.prop_cycler)["color"]
        plt.plot(self.times_sq, self.sigma_y_sq, "o", color=color, label="Y")
        plt.plot(
            self.times_sq, self.y_fit[0] * self.times_sq + self.y_fit[1], color=color
        )
        plt.set_xlabel("t^2 (ms^2)")
        plt.set_ylabel("sigma^2 (mm^2)")

        logging.info("Temp. X: %.4g\tTemp. Y: %.4g", self.x_temp, self.y_temp)

        plt.legend()

        return fig
