import logging

from collections import deque
from abc import ABC, abstractmethod

import numpy as np

from boltons.cacheutils import cachedproperty
from scipy.stats import linregress

from config import config


class ShotSequence(ABC):
    def __init__(self, independent_var):
        self.independent_var = np.array(independent_var)
        self.shots = deque(maxlen=len(self.independent_var))

    @abstractmethod
    def x(self):
        pass

    @abstractmethod
    def y(self):
        pass

    @property
    def sigma_x(self):
        return self.shot_param_arr("sx") * config.physical_scale

    @property
    def sigma_y(self):
        return self.shot_param_arr("sy") * config.physical_scale

    @property
    def atom_number(self):
        return np.array([s.atom_number for s in self.shots])

    def shot_param_arr(self, key):
        return np.array([s.fit.best_values[key] for s in self.shots])

    def add(self, shot, cb):
        if len(self.shots) < self.shots.maxlen:
            logging.info(
                "Added shot %s as x=%.2f to sequence!",
                shot.name,
                self.independent_var[len(self.shots)],
            )
            self.shots.append(shot)
            if len(self.shots) == self.shots.maxlen:
                cb(self)

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def plot(self, fig):
        pass


class TimeOfFlight(ShotSequence):
    KB = 1.38e-23

    @property
    def x(self):
        return np.square(self.independent_var + config.repump_time)

    @property
    def y(self):
        return ("X", np.square(self.sigma_x)), ("Y", np.square(self.sigma_y))

    @cachedproperty
    def fit(self):
        results = []
        for label, y in self.y:
            fit = linregress(self.x, y)
            results.append((label, y, fit))

        return results

    @property
    def temperatures(self):
        return [f[0] * config.atom_mass / type(self).KB * 1e6 for _, _, f in self.fit]

    @property
    def temperature_errors(self):
        return [f[4] * config.atom_mass / type(self).KB * 1e6 for _, _, f in self.fit]

    @property
    def avg_temp(self):
        return np.mean(self.temperatures)

    @property
    def avg_temp_err(self):
        return 0.5 * np.sqrt(np.sum(np.square(self.temperature_errors)))

    def plot(self, fig):
        fig.clf()
        plt = fig.add_subplot()

        for label, y, fit in self.fit:
            color = next(plt._get_lines.prop_cycler)["color"]
            plt.plot(self.x, y, "o", color=color, label=label)
            plt.plot(self.x, fit[0] * self.x + fit[1], color=color)

        plt.set_xlabel("t^2 (ms^2)")
        plt.set_ylabel("sigma^2 (mm^2)")

        logging.info(
            "Temp. X: %.4g\tTemp. Y: %.4g", self.temperatures[0], self.temperatures[1]
        )

        plt.legend()

        return fig


class AtomNumberOptimization(ShotSequence):
    @property
    def x(self):
        return self.independent_var

    @property
    def y(self):
        return ("Atom Number", self.atom_number)

    def fit(self):
        pass

    def plot(self, fig):
        fig.clf()
        plt = fig.add_subplot()

        plt.plot(self.x, self.y[1], "o", label=self.y[0])

        plt.set_ylabel(self.y[0])

        return fig
