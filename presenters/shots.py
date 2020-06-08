import logging
import os.path
from os import path

from pathlib import Path
from datetime import date
from collections import deque
from threading import Lock

import h5py
import time
from matplotlib.figure import Figure

from utils.threading import mainthread

from config import config
from models import shots

class ShotPresenter:
    def __init__(self, app, worker, *, plot_view, fit_view, list_view, threeroi_view, settings_view):
        self.app = app
        self.worker = worker
        self.plot_view = plot_view
        self.fit_view = fit_view
        self.list_view = list_view
        self.threeroi_view = threeroi_view
        self.settings_view = settings_view
        # Stores data for past (maxlen) shots
        self.recent_shots = deque(maxlen=15)
        self.current_shot = None
        self.shotlist_selection = ()

        self.recent_shots_lock = Lock()

    ##### Non-GUI methods #####
    def process_shot(self, name, paths):
        logging.info("\n-------------------------------")
        logging.info("1: PROCESSING SHOT %s", name)

        shot = shots.Shot(name, paths)
        self.current_shot = shot
        self._update_recent_shots(shot)

        self.app.queue(self.display_shot, shot)  # Display raw aborption image

        if config.fit:
            shot.run_fit(config)
            self.app.queue(self.display_shot, shot)  # Display fit overlay

        # Save to png output
        figure = Figure(figsize=(8, 5))
        shot.plot(figure)
        figure.savefig(_output_path(name), dpi=150)

        # Saves fit params to log file
        cmnts = self.settings_view.get_comment()
        logging.info("Updating logging.csv for shot %s with comment %s " % (name, cmnts))

        # Checks if log file already exists, if not creates a new one
        with h5py.File(_output_log_path(name), "a") as logfile:
            lf = logfile.create_group(str(time.strftime('%H:%M:%S'))) # creates group based on 24hr timestamp
            lf.create_dataset("atom", data = shot.data)
            lf.create_dataset("beam", data = shot.beam)
            lf.create_dataset("dark", data = shot.dark)
            lf.attrs['filename'] = str(name)
            lf.attrs['atom_number'] = shot.atom_number

            for label, value in config.logdict.items(): # Appends config snapshot
                lf.attrs[label] = value

            if config.roi_enabled: # only appends if roi is enabled
                lf.attrs["roi"] = config.roi
                lf.attrs['fit_vars'] = shot.fit.best_values

            if config.three_roi_enabled: # only appends value if threeroi is enabled
                lf.attrs['a_b_ratio'] = shot.three_roi_atom_number["a_b_ratio"]
                lf.attrs["threeroi"] = config.threeroi

            lf.attrs["comments"] = cmnts

        # Check if ToF or optimization
        self.app.sequence_presenter.add_shot(shot)

    def _update_recent_shots(self, shot):
        with self.recent_shots_lock:
            if shot in self.recent_shots:
                idx = self.recent_shots.index(shot)
                self.recent_shots.remove(shot)
                self.recent_shots.insert(idx, shot)
            else:
                self.recent_shots.append(shot)

    ##### GUI methods #####
    @mainthread
    def display_shot(self, shot):
        """Updates the data display with new info."""
        self.plot_view.display(shot)
        # TODO: plotting is done in the main thread - any way to pass figure in?

        # Update fit params
        self.fit_view.clear()
        if shot.fit:
            self.fit_view.display(dict({"N": shot.atom_number}, **shot.fit.best_values))
        else:
            self.fit_view.display({"N": shot.atom_number})

        # Synchronize recent shots view with data
        with self.recent_shots_lock:
            self.list_view.refresh(self.recent_shots)
            self.list_view.focus(shot)

        if config.three_roi_enabled:
            #self.threeroi_view.threeroi_counts = shot.three_roi_atom_number
            self.threeroi_view.display(shot.three_roi_atom_number) 

    @mainthread
    def display_recent_shot(self, idx):
        """Displays a shot in the recent_shot deque by idx."""
        self.display_shot(self.recent_shots[idx])

    def update_shotlist_selection(self, indexes):
        self.shotlist_selection = tuple(self.recent_shots[idx] for idx in indexes)

    def refit_current_shot(self):
        self.worker.submit(self.refit_shot, self.current_shot)

    def refit_shot(self, shot):
        shot.clear_fit()
        self.app.queue(self.display_shot, shot)

        if config.fit:
            shot.run_fit(config)
            self.app.queue(self.display_shot, shot)



def _output_path(name):
    """Move processed images to "Analysis Results" folder by date"""
    output = Path("../Analysis Results/").joinpath(str(date.today()))
    output.mkdir(parents=True, exist_ok=True)
    return output.joinpath(f"{name}.png")

def _output_log_path(name):
    """Sets the path directory for generating a log file in hdf5 format in the raw data folder"""
    output = Path("../Raw Data/").joinpath(str(date.today()))
    output.mkdir(parents=True, exist_ok=True)
    return output.joinpath("logging.hdf5")
