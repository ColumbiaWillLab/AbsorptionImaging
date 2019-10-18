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
        self.current_atom_opt = None

    @mainthread
    def display_tof(self, tof):
        self.plot_view.display(tof)
        self.fit_view.display(tof)

    @mainthread
    def display_atom_opt(self, ao):
        self.plot_view.display(ao)

    def start_tof(self, times):
        self.current_tof = sequences.TimeOfFlight(times)
        logging.info("Starting Time of Flight: %s", str(times))

    def start_atom_opt(self, params):
        self.current_atom_opt = sequences.AtomNumberOptimization(params)
        logging.info("Starting Atom Number Optimization: %s", str(params))

    def start_tof_selection(self, times):
        selection = self._get_shot_selection(times)

        if selection:
            self.start_tof(times)
            for shot in selection:
                self.current_tof.add(shot, self._sequence_complete)

    def start_atom_opt_selection(self, params):
        selection = self._get_shot_selection(params)

        if selection:
            self.start_atom_opt(params)
            for shot in selection:
                self.current_atom_opt.add(shot, self._sequence_complete)

    def add_shot(self, shot):
        if self.current_tof:
            self.current_tof.add(shot, self._sequence_complete)

        if self.current_atom_opt:
            self.current_atom_opt.add(shot, self._sequence_complete)

    def _sequence_complete(self, sequence):
        if isinstance(sequence, sequences.TimeOfFlight):
            self.app.queue(self.display_tof, sequence)
        elif isinstance(sequence, sequences.AtomNumberOptimization):
            self.app.queue(self.display_atom_opt, sequence)

    def _get_shot_selection(self, params):
        selection = self.app.shot_presenter.shotlist_selection

        if len(selection) != len(params):
            logging.error(
                "Sequence mismatch! %i selected for %i params",
                len(selection),
                len(params),
            )
            return False
        return selection

    def _autofill(self, params):
        logging.info("Autofilled String: %s", str(params))
