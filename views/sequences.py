import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as tkst

from math import isfinite

import numpy as np

from config import config

from .components import FloatEntry


class ToFFit(ttk.Frame):
    def __init__(self, master, presenter):
        self.master = master
        self.presenter = presenter

        super().__init__(self.master)

        # Left Frame (ToF Controls)
        sequence_params = SequenceParams(
            self, self.presenter, run="start_tof", fit_selected="start_tof_selection"
        )
        sequence_params.pack(side="left", expand=True, anchor="e", pady=15, padx=15)

        # Right Frame
        right_frame = ttk.Frame(self)
        right_frame.pack(side="left", expand=True, anchor="w", pady=15, padx=15)

        # Right Top Frame (Temperature Output)
        rt_frame = ttk.Frame(right_frame)
        rt_frame.pack(side="top")

        self.temp_entries = []
        r = 0
        ttk.Label(rt_frame, text="Temperature (µK)").grid(row=r, column=1)
        ttk.Label(rt_frame, text="Std. Error (µK)").grid(row=r, column=2)
        for i, text in enumerate(("X", "Y", "Avg.")):
            r = i + 1
            ttk.Label(rt_frame, text=text).grid(row=r, column=0)

            temp = ttk.Entry(rt_frame, state="readonly", width=10)
            err = ttk.Entry(rt_frame, state="readonly", width=10)
            temp.grid(row=r, column=1)
            err.grid(row=r, column=2)
            self.temp_entries.append((temp, err))

        r += 1
        ttk.Label(rt_frame, text="Mean Atom #").grid(row=r, column=1, pady=((10, 0)))
        ttk.Label(rt_frame, text="Coeff. Var. (%)").grid(
            row=r, column=2, pady=((10, 0))
        )

        r += 1
        atom_n_mean = ttk.Entry(rt_frame, state="readonly", width=10)
        atom_n_mean.grid(row=r, column=1)
        atom_n_cv = ttk.Entry(rt_frame, state="readonly", width=10)
        atom_n_cv.grid(row=r, column=2)

        self.atom_n_mean = atom_n_mean
        self.atom_n_cv = atom_n_cv

        # Right Bottom Frame (Config)
        rb_frame = ttk.Frame(right_frame)
        rb_frame.pack(side="top", pady=(50, 0))

        ttk.Label(rb_frame, text="Repump Time (ms)").grid(row=0, column=0)
        ttk.Label(rb_frame, text="Mass (kg)").grid(row=1, column=0)

        repump_entry = FloatEntry(rb_frame, state="normal", width=10)
        repump_entry.insert(0, config.repump_time)
        repump_entry.grid(row=0, column=1)

        mass_entry = FloatEntry(rb_frame, state="normal", width=10)
        mass_entry.insert(0, config.atom_mass)
        mass_entry.grid(row=1, column=1)

        self.repump_entry = repump_entry
        self.mass_entry = mass_entry

    def display(self, tof):
        atom_n_mean = np.mean(tof.atom_number)
        pairs = [
            (self.temp_entries[0][0], tof.temperatures[0]),
            (self.temp_entries[1][0], tof.temperatures[1]),
            (self.temp_entries[0][1], tof.temperature_errors[0]),
            (self.temp_entries[1][1], tof.temperature_errors[1]),
            (self.temp_entries[2][0], tof.avg_temp),
            (self.temp_entries[2][1], tof.avg_temp_err),
            (self.atom_n_mean, atom_n_mean),
            (self.atom_n_cv, np.std(tof.atom_number) / atom_n_mean * 100),
        ]
        for entry, value in pairs:
            entry.configure(state="normal")
            entry.delete(0, "end")
            entry.insert(0, "{:.4g}".format(value))
            entry.configure(state="readonly")


class AtomNumberOptimization(ttk.Frame):
    def __init__(self, master, presenter):
        self.master = master
        self.presenter = presenter

        super().__init__(self.master)

        # Left Frame (ToF Controls)
        sequence_params = SequenceParams(
            self,
            self.presenter,
            run="start_atom_opt",
            fit_selected="start_atom_opt_selection",
        )
        sequence_params.pack(expand=True)


class SequenceParams(ttk.Frame):
    def __init__(self, master, presenter, *, run, fit_selected):
        self.master = master
        self.presenter = presenter
        self.run = run
        self.fit_selected = fit_selected
        self.autofill = autofill

        super().__init__(self.master)

        ttk.Label(self, text="Parameter:").grid(row=0, column=0)
        ttk.Label(self, text="Autofill:").grid(row=0, column=1)

        st = tkst.ScrolledText(self, state="normal", height=10, width=10)
        st.grid(row=1,column=0,pady=(0, 5),rowspan=5)

        run_btn = ttk.Button(self, text="Run", command=self._run)
        run_btn.grid(row=6,column=0)

        fit_selected = ttk.Button(self, text="Fit Selected", command=self._fit_selected)
        fit_selected.grid(row=7,column=0)

        ttk.Label(self, text="Start:").grid(row=1, column=1)
        autofillstart = ttk.Entry(self,width=5)
        autofillstart.grid(row=2,column=1)

        ttk.Label(self, text="End:").grid(row=3, column=1)
        autofillend = ttk.Entry(self,width=5)
        autofillend.grid(row=4,column=1)

        ttk.Label(self, text="Step:").grid(row=5, column=1)
        autofillstep = ttk.Entry(self,width=5)
        autofillstep.grid(row=6,column=1)

        autofill_btn = ttk.Button(self, text="Autofill", command=self._autofill)
        autofill_btn.grid(row=7,column=1)

        self.st = st

    def _check_sequence(self):
        try:
            vals = [float(x) for x in self.st.get("1.0", "end-1c").split()]
            if not all(x >= 0 and isfinite(x) for x in vals):
                raise ValueError

            return vals
        except ValueError:
            tk.messagebox.showerror("Atom Number Optimization", "Invalid Entry!")

    def _run(self):
        vals = self._check_sequence()
        getattr(self.presenter.sequence_presenter, self.run)(vals)

    def _fit_selected(self):
        vals = self._check_sequence()
        getattr(self.presenter.sequence_presenter, self.fit_selected)(vals)

    def _autofill(self):
        """Autofills scrolled text based on input start,end,step variables"""
        afstart = float(autofillstart.get())
        afend = float(autofillend.get())
        afstep = float(autofillstep.get())
        aflist = np.arange(afstart,afend,afstep)
        for x in aflist:
            st.insert(tk.INSERT,x)
            st.insert(tk.INSERT,'\n')
