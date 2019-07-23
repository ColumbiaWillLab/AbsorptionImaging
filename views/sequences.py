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
        left_frame = ttk.Frame(self)
        left_frame.grid(row=0, column=0, rowspan=2)

        ttk.Label(left_frame, text="Times of Flight:").pack()

        st = tkst.ScrolledText(left_frame, state="normal", height=12, width=10)
        st.pack(pady=(0, 5))
        if config.tof:
            st.insert("1.0", "\n".join(map(str, config.tof)))
        else:
            st.insert("1.0", "0.25\n0.5\n0.75\n1\n1.25\n1.5\n1.75\n2\n2.25\n2.5")

        run_btn = ttk.Button(left_frame, text="Run", command=self._run)
        run_btn.pack(anchor="w")

        fit_selected = ttk.Button(
            left_frame, text="Fit Selected", command=self._fit_selected
        )
        fit_selected.pack()

        self.st = st
        self.run_btn = run_btn
        self.fit_selected = fit_selected

        # Right Top Frame (Temperature Output)
        rt_frame = ttk.Frame(self)
        rt_frame.grid(row=0, column=1)

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
        rb_frame = ttk.LabelFrame(self)
        rb_frame.grid(row=1, column=1, pady=(10, 0))

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

    def _check_tof(self):
        config["atoms"]["repump_time"] = str(self.repump_entry.get())
        config["atoms"]["mass"] = str(self.mass_entry.get())
        config.save()

        try:
            vals = [float(x) for x in self.st.get("1.0", "end-1c").split()]
            if not all(x >= 0 and isfinite(x) for x in vals):
                raise ValueError

            config.tof = vals
            config.save()
            return vals
        except ValueError:
            tk.messagebox.showerror("Temperature Fitting Alert", "Invalid Entry!")

    def _run(self):
        vals = self._check_tof()
        self.presenter.sequence_presenter.start_tof(vals)

    def _fit_selected(self):
        vals = self._check_tof()
        self.presenter.sequence_presenter.start_tof_selection(vals)

    def display(self, tof):
        atom_n_mean = np.mean(tof.atom_number)
        pairs = [
            (self.temp_entries[0][0], tof.x_temp),
            (self.temp_entries[1][0], tof.y_temp),
            (self.temp_entries[0][1], tof.x_temp_err),
            (self.temp_entries[1][1], tof.y_temp_err),
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
