import logging

from math import isfinite

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as tkst

import numpy as np

from config import config
from watcher import observer
from queues import shot_list

from .components import FloatEntry


class FitParams(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.fit_params = {}
        self.config_params = {}

        # Fit Parameters (uneditable)
        params_frame = ttk.Frame(self)
        params_frame.pack(side="left", fill="y", expand=True)
        keys = ["N", "A", "x0", "y0", "sx", "sy", "theta", "z0"]
        labels = ["N", "A", "x_0", "y_0", "σ_x", "σ_y", "θ", "z_0"]
        for l_idx, lbl in enumerate(labels):
            ttk.Label(params_frame, text=lbl).grid(row=l_idx, column=0)

        for f_idx in range(8):
            entry = ttk.Entry(params_frame, state="readonly")
            entry.grid(row=f_idx, column=1)
            self.fit_params[keys[f_idx]] = entry

        options_frame = ttk.Frame(self)
        options_frame.pack(side="left", fill="y", expand=True)

        # Region of Interest Control
        roi_frame = ttk.LabelFrame(options_frame, text="ROI")
        roi_frame.pack(fill="x", expand=True)
        ttk.Label(roi_frame, text="X").grid(row=0, column=1)
        ttk.Label(roi_frame, text="Y").grid(row=0, column=2)

        ttk.Label(roi_frame, text="Top Left").grid(row=1, column=0)
        self.roi_tl_x = FloatEntry(roi_frame, width=4)
        self.roi_tl_y = FloatEntry(roi_frame, width=4)
        self.roi_tl_x.grid(row=1, column=1)
        self.roi_tl_y.grid(row=1, column=2)

        ttk.Label(roi_frame, text="Bottom Right").grid(row=2, column=0)
        self.roi_br_x = FloatEntry(roi_frame, width=4)
        self.roi_br_y = FloatEntry(roi_frame, width=4)
        self.roi_br_x.grid(row=2, column=1)
        self.roi_br_y.grid(row=2, column=2)

        self.roi_entries = [self.roi_tl_x, self.roi_tl_y, self.roi_br_x, self.roi_br_y]
        for entry in self.roi_entries:
            entry.bind("<Return>", self._update_roi)
        if config.roi:
            for i, entry in enumerate(self.roi_entries):
                entry.delete(0, "end")
                entry.insert(0, config.roi[i])

        self.toggle_roi = ttk.Button(roi_frame, text="Enable", command=self._toggle_roi)
        self.toggle_roi.grid(row=3, column=1, columnspan=2)

        # Center Control
        self.fixcenter = tk.BooleanVar()
        self.fixcenter.set(config.fix_center)
        fixcenter_btn = ttk.Checkbutton(
            roi_frame,
            text="Fix Center",
            variable=self.fixcenter,
            command=self._toggle_center,
        )
        fixcenter_btn.grid(row=4, column=0, pady=(20, 0))

        center_x_var = tk.DoubleVar()
        self.center_x = FloatEntry(roi_frame, width=4, textvariable=center_x_var)
        self.center_x.var = center_x_var

        center_y_var = tk.DoubleVar()
        self.center_y = FloatEntry(roi_frame, width=4, textvariable=center_y_var)
        self.center_y.var = center_y_var

        if config.center:
            x, y = config.center
            self.center_x.var.set(x)
            self.center_y.var.set(y)

        self.center_x.grid(row=4, column=1, pady=(20, 0))
        self.center_y.grid(row=4, column=2, pady=(20, 0))

        self.center_x.bind("<Return>", self._update_center)
        self.center_y.bind("<Return>", self._update_center)

        self.last_shot_center = ttk.Button(
            roi_frame, text="Last Shot", command=self._fill_last_shot_center
        )
        self.last_shot_center.grid(row=5, column=1, columnspan=2)

        fit_frame = ttk.LabelFrame(options_frame, text="Fit")
        fit_frame.pack(fill="x", expand=True)
        self.fixtheta = tk.BooleanVar()
        self.fixtheta.set(config.fix_theta)
        fixtheta_btn = ttk.Checkbutton(
            fit_frame,
            text="Fix Theta",
            variable=self.fixtheta,
            command=self._toggle_fixtheta,
        )
        fixtheta_btn.grid(row=0, column=0, sticky="w")
        self.fitvar = tk.BooleanVar()
        self.fitvar.set(config.fit)
        fitbtn = ttk.Checkbutton(
            fit_frame,
            text="Enable Fitting",
            variable=self.fitvar,
            command=self._toggle_fit,
        )
        fitbtn.grid(row=1, column=0, sticky="w")

    @property
    def keys(self):
        return ["N", "A", "x0", "y0", "sx", "sy", "theta", "z0"]

    def display(self, fit_params):
        for k, v in fit_params.items():
            if k in ["x0", "y0", "sx", "sy"]:
                v *= config.pixel_size
            elif k == "theta":
                v = np.degrees(v)
            entry = self.fit_params[k]
            entry.configure(state="normal")
            entry.delete(0, "end")
            entry.insert(0, "{:.4g}".format(v))
            entry.configure(state="readonly")

    def clear(self):
        for entry in self.fit_params.values():
            entry.configure(state="normal")
            entry.delete(0, "end")
            entry.configure(state="readonly")

    def _toggle_fixtheta(self):
        config["fit"]["fix_theta"] = str(self.fixtheta.get())
        config.save()

    def _toggle_fit(self):
        config.fit = self.fitvar.get()

    def _update_roi(self, event=None):
        try:
            roi = tuple(int(v.get()) for v in self.roi_entries)
        except ValueError:
            logging.error("Malformed ROI params!")
            return False

        if roi[0] < roi[2] and roi[1] < roi[3]:
            config.roi = roi
            config.save()
            logging.info("Updated region of interest: %s", str(roi))
            return True
        else:
            logging.warning("Invalid ROI params!")

    def _toggle_roi(self):
        if config.roi_enabled:
            config.roi_enabled = False
            self.toggle_roi.configure(text="Enable", state="normal")
            logging.debug("ROI disabled")
        else:
            if self._update_roi():
                config.roi_enabled = True
                self.toggle_roi.configure(text="Disable", state="active")
                logging.debug("ROI enabled")

    def _update_center(self, event=None):
        x, y = self.center_x.var.get(), self.center_y.var.get()
        if not x or not y:
            return False
        try:
            center = tuple((x, y))
        except ValueError:
            logging.error("Malformed center fix params!")
            return False

        config.center = center
        config.save()
        logging.info("Updated center fix: %s", str(center))
        return True

    def _toggle_center(self):
        config.fix_center = self.fixcenter.get()
        if config.fix_center:
            self._update_center()

    def _fill_last_shot_center(self):
        try:
            shot = shot_list[-1]
            if "twoD_gaussian" in shot.__dict__:
                self.center_x.var.set(shot.twoD_gaussian.best_values["x0"])
                self.center_y.var.set(shot.twoD_gaussian.best_values["y0"])
                self._update_center()
            else:
                logging.warning("Shot has no 2D Gaussian fit.")
        except IndexError:
            logging.error("No last shot to pull from!")


class TemperatureParams(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        # Left Frame (ToF Controls)
        left_frame = ttk.Frame(self)
        left_frame.grid(row=0, column=0, rowspan=2)

        ttk.Label(left_frame, text="Times of Flight:").pack()

        st = tkst.ScrolledText(left_frame, state="normal", height=12, width=10)
        st.pack()
        if config.tof:
            st.insert("1.0", "\n".join(map(str, config.tof)))
        else:
            st.insert("1.0", "0.25\n0.5\n0.75\n1\n1.25\n1.5\n1.75\n2\n2.25\n2.5")

        btn = ttk.Button(left_frame, text="Run", command=self.run)
        btn.pack()

        self.st = st
        self.btn = btn

        # Right Top Frame (Temperature Output)
        rt_frame = ttk.Frame(self)
        rt_frame.grid(row=0, column=1)

        self.temp_entries = []
        for i, text in enumerate(("X Temp.", "Y Temp.", "Avg. Temp")):
            ttk.Label(rt_frame, text=text).grid(row=i, column=0)
            ttk.Label(rt_frame, text="Std. Error").grid(row=i, column=2)
            temp = ttk.Entry(rt_frame, state="readonly", width=10)
            err = ttk.Entry(rt_frame, state="readonly", width=10)
            temp.grid(row=i, column=1)
            err.grid(row=i, column=3)
            self.temp_entries.append((temp, err))

        ttk.Label(rt_frame, text="Mean Atom #").grid(row=i + 1, column=0)
        atom_n_mean = ttk.Entry(rt_frame, state="readonly", width=10)
        atom_n_mean.grid(row=i + 1, column=1)

        ttk.Label(rt_frame, text="Coeff. Var.").grid(row=i + 1, column=2)
        atom_n_cv = ttk.Entry(rt_frame, state="readonly", width=10)
        atom_n_cv.grid(row=i + 1, column=3)

        self.atom_n_mean = atom_n_mean
        self.atom_n_cv = atom_n_cv

        # Right Bottom Frame (Config)
        rb_frame = ttk.Frame(self)
        rb_frame.grid(row=1, column=1)

        ttk.Label(rb_frame, text="Repump Time").grid(row=0, column=0)
        ttk.Label(rb_frame, text="Mass").grid(row=1, column=0)

        repump_entry = FloatEntry(rb_frame, state="normal", width=10)
        repump_entry.insert(0, config.repump_time)
        repump_entry.grid(row=0, column=1)

        mass_entry = FloatEntry(rb_frame, state="normal", width=10)
        mass_entry.insert(0, config.atom_mass)
        mass_entry.grid(row=1, column=1)

        self.repump_entry = repump_entry
        self.mass_entry = mass_entry

    def run(self):
        config["atoms"]["repump_time"] = str(self.repump_entry.get())
        config["atoms"]["mass"] = str(self.mass_entry.get())
        config.save()

        try:
            vals = [float(x) for x in self.st.get("1.0", "end-1c").split()]
            if not all(x >= 0 and isfinite(x) for x in vals):
                raise ValueError

            observer.start_tof(vals)
            config.tof = vals
            config.save()
        except ValueError:
            tk.messagebox.showerror("Temperature Fitting Alert", "Invalid Entry!")

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


class ExperimentParams(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.config_params = {}

        p_idx = 0
        for section in ("camera", "beam"):
            for key in config[section].keys():
                text = key.replace("_", " ").capitalize()
                ttk.Label(self, text=text).grid(row=p_idx, column=0)
                entry = FloatEntry(self, state="normal")
                entry.grid(row=p_idx, column=1)
                entry.insert(0, config[section].getfloat(key))
                self.config_params[f"{section}.{key}"] = entry
                p_idx += 1

        save = ttk.Button(self, text="Save", command=self._save_config)
        save.grid(row=p_idx, column=1)

    def _save_config(self):
        for name, entry in self.config_params.items():
            section, key = name.split(".")
            config[section][key] = str(entry.get())
        config.save()


class PlotSettings(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.colormap = tk.StringVar(
            self, name="colormap", value=config.get("plot", "colormap")
        )
        color_options = ("cividis", "viridis", "Greys")
        ttk.Label(self, text="Colormap").grid(row=0, column=0)
        ttk.OptionMenu(
            self,
            self.colormap,
            self.colormap.get(),
            *color_options,
            command=lambda val: self.save_config("colormap", val),
        ).grid(row=0, column=1)

    def save_config(self, name, val):
        config.set("plot", name, value=val)
        config.save()
