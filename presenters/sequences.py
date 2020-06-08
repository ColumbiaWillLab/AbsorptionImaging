import logging
import os.path
from os import path

import h5py
import time
from pathlib import Path
from datetime import date
from utils.threading import mainthread

from models import sequences
from config import config

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
        # Updating GUI display
        self.plot_view.display(tof)
        self.fit_view.display(tof)

    @mainthread
    def display_atom_opt(self, ao):
        self.plot_view.display(ao)

    def start_tof(self, times):
        self.current_tof = sequences.TimeOfFlight(times)
        logging.info("Starting Time of Flight (ms): %s", str(times))

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

            # Saving tof params into logging.hdf5
            logging.info("Updating logging.hdf5 with tof params...")

            # Appends sequence information as a subgroup tof optimization
            with h5py.File(self._output_log_path(), "a") as logfile:
                lf = logfile.create_group("/tof_sequence/" + str(time.strftime('%H:%M:%S'))) # Sets name to be timestamp in subgroup
                lf.attrs['filename'] = str([shot.name for shot in sequence.shots])
                lf.create_dataset("atom_number", data = sequence.atom_number)
                lf.create_dataset("time_sequence", data = sequence.t)
                lf.attrs['average_T(uK)'] = str(sequence.avg_temp)

                for label, value in config.logdict.items(): # Appends config snapshot
                    lf.attrs[label] = value


        elif isinstance(sequence, sequences.AtomNumberOptimization):
            self.app.queue(self.display_atom_opt, sequence)

            # Saving tof params into logging.hdf5
            logging.info("Updating logging.hdf5 with tof params...")

            # Appends sequence information as a subgroup for atom num optimization
            with h5py.File(_output_log_path(), "a") as logfile:
                lf = logfile.create_group("/atomnum_sequence/" + str(time.strftime('%H:%M:%S'))) # Sets name to be timestamp in group
                lf.attrs['filename'] = str([shot.name for shot in sequence.shots])
                lf.create_dataset("atom_number", data = sequence.atom_number)

                for label, value in config.logdict.items(): # Appends config snapshot
                    lf.attrs[label] = value

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

    def _output_log_path(self):
        """Sets the path directory for generating a log file in the raw data folder"""
        output = Path("../Raw Data/").joinpath(str(date.today()))
        output.mkdir(parents=True, exist_ok=True)
        return output.joinpath("logging.hdf5")
