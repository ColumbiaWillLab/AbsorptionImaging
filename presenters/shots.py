import logging
import csv
import os.path
from os import path

from pathlib import Path
from datetime import date
from collections import deque
from threading import Lock

from matplotlib.figure import Figure

from utils.threading import mainthread

from config import config
from models import shots


class ShotPresenter:
    def __init__(self, app, worker, *, plot_view, fit_view, list_view, threeroi_view):
        self.app = app
        self.worker = worker
        self.plot_view = plot_view
        self.fit_view = fit_view
        self.list_view = list_view
        self.threeroi_view = threeroi_view
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
        logging.info("Updating log for shot %s ", name)


        # Checks if log file already exists, if not creates a new one
        if not path.exists(_output_log_path(name)):
            with open(_output_log_path(name), 'w', newline='') as logfile:
                writer = csv.DictWriter(logfile, fieldnames = config.logheader) # Pulls headers from config.ini
                writer.writeheader()

        # Appends requisite data
        with open(_output_log_path(name), 'a', newline='') as logfile:
            writer = csv.DictWriter(logfile, fieldnames = config.logheader) # Pulls headers from config.ini
            writer.writerow({"filename" : name,
                             "magnification" : config.magnification,
                             "atom number" : shot.atom_number,
                             "fitted shot" : config.fit})

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
    """Sets the path directory for generating a log file in the raw data folder"""
    output = Path("../Raw Data/").joinpath(str(date.today()))
    output.mkdir(parents=True, exist_ok=True)
    return output.joinpath("logging.csv")
