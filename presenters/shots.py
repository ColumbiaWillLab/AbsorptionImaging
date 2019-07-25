import logging

from pathlib import Path
from datetime import date
from collections import deque

from matplotlib.figure import Figure

from utils.threading import mainthread

from config import config
from models import shots


class ShotPresenter:
    def __init__(self, app, worker, *, plot_view, fit_view, list_view):
        self.app = app
        self.worker = worker
        self.plot_view = plot_view
        self.fit_view = fit_view
        self.list_view = list_view

        self.recent_shots = deque(maxlen=15)
        self.current_shot = None
        self.shotlist_selection = ()

    ##### Non-GUI methods #####
    def process_shot(self, name, paths):
        logging.info("\n-------------------------------")
        logging.info("1: PROCESSING SHOT %s", name)

        shot = shots.Shot(name, paths)
        self.current_shot = shot
        self._update_recent_shots(shot)

        self.app.queue(self.display_shot, shot)  # Display raw aborption image

        if config.fit:
            shot.fit_2D(config)
            self.app.queue(self.display_shot, shot)  # Display fit overlay

        figure = Figure(figsize=(8, 5))
        figure.savefig(_output_path(name), dpi=150)

        self.app.sequence_presenter.add_shot(shot)

    def _update_recent_shots(self, shot):
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

        self.fit_view.clear()
        if shot.fit:
            self.fit_view.display(
                dict({"N": shot.atom_number}, **shot.fit.result.best_values)
            )
        else:
            self.fit_view.display({"N": shot.atom_number})

        self.list_view.refresh(self.recent_shots)
        self.list_view.focus(shot)

    @mainthread
    def display_recent_shot(self, idx):
        """Displays a shot in the recent_shot deque by idx."""
        self.display_shot(self.recent_shots[idx])

    def update_shotlist_selection(self, indexes):
        self.shotlist_selection = tuple(self.recent_shots[idx] for idx in indexes)

    def refit_current_shot(self):
        self.worker.submit(self.refit_shot, self.current_shot)

    def refit_shot(self, shot):
        shot.fit_2D(config)
        self.app(self.display_shot, shot)


def _output_path(name):
    """Move processed images to "Analysis Results" folder by date"""
    output = Path("../Analysis Results/").joinpath(str(date.today()))
    output.mkdir(parents=True, exist_ok=True)
    return output.joinpath(f"{name}.png")
