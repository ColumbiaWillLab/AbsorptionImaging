import logging

from utils.threading import mainthread

from models import sequences


class SequencePresenter:
    def __init__(self, app, worker, *, plot_view, fit_view):
        self.app = app
        self.worker = worker
        self.plot_view = plot_view
        self.fit_view = fit_view

        self.current_tof = None

    @mainthread
    def display_tof(self, tof):
        self.plot_view.display(tof)
        self.fit_view.display(tof)

    def start_tof(self, times):
        self.current_tof = sequences.TimeOfFlight(times)
        logging.info("Starting Time of Flight: %s", str(times))

    def start_tof_selection(self, times):
        selection = self.app.shot_presenter.shotlist_selection

        if len(selection) != len(times):
            logging.error(
                "ToF shot mismatch! %i selected for %i times",
                len(selection),
                len(times),
            )
        else:
            current_tof = sequences.TimeOfFlight(times)
            logging.info("Starting Time of Flight: %s", str(times))
            for shot in selection:
                current_tof.add(shot, self._tof_complete)

    def add_shot_to_tof(self, shot):
        if self.current_tof:
            self.current_tof.add(shot, self._tof_complete)

    def _tof_complete(self, tof):
        self.app.queue(self.display_tof, tof)
