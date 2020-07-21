import logging
import tkinter as tk
import tkinter.ttk as ttk

import numpy as np

from config import config

from .components import FloatEntry
from itertools import chain


class ShotList(ttk.Frame):
    def __init__(self, master, presenter, **kw):
        self.presenter = presenter
        self.master = master

        super().__init__(master)
        self.pack(fill="both", expand=True)

        kw["columns"] = ("atoms", "sigma_x", "sigma_y")
        kw["selectmode"] = "extended"
        self.tree = ttk.Treeview(self, **kw)
        self.tree.pack(side="left", fill="both", expand=True)

        # Scrollbar
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        vsb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.column("#0", anchor="w", width=200)
        self.tree.column("atoms", anchor="w", width=100, stretch=False)
        self.tree.column("sigma_x", anchor="w", width=100, stretch=False)
        self.tree.column("sigma_y", anchor="w", width=100, stretch=False)

        self.tree.heading("#0", text="Shot")
        self.tree.heading("atoms", text="Atom Number")
        self.tree.heading("sigma_x", text="Std. Dev. X")
        self.tree.heading("sigma_y", text="Std. Dev. Y")

        self.tree.bind("<Double-1>", self._on_double_click)
        self.tree.bind("<Return>", self._on_return_keypress)

        self.tree.bind("<<TreeviewSelect>>", self._on_treeview_select)

    def refresh(self, shots):
        self.clear()

        for shot in shots:
            if shot.fit:
                values = (
                    shot.atom_number,
                    shot.fit.best_values["sx"] * config.pixel_size,
                    shot.fit.best_values["sy"] * config.pixel_size,
                )
            else:
                values = (shot.atom_number,)
            self.tree.insert(
                "",
                "end",
                id=shot.name,
                text=shot.name,
                values=tuple(map("{:.4g}".format, values)),
            )

    def clear(self):
        self.tree.delete(*self.tree.get_children())

    def focus(self, shot):
        self.tree.selection_set(shot.name)
        self.tree.focus(shot.name)
        self.tree.see(shot.name)

    def _on_double_click(self, event):
        idx = self.tree.index(self.tree.identify("item", event.x, event.y))
        self.presenter.shot_presenter.display_recent_shot(idx)

    def _on_return_keypress(self, event):
        idx = self.tree.index(self.tree.focus())
        self.presenter.shot_presenter.display_recent_shot(idx)

    def _on_treeview_select(self, event):
        num_selected = len(self.tree.selection())

        if num_selected > 1:
            self.master.configure(text="Shots (selected: %i)" % num_selected)
        else:
            self.master.configure(text="Shots")

        indexes = [self.tree.index(s) for s in self.tree.selection()]
        self.presenter.shot_presenter.update_shotlist_selection(indexes)


class ShotFit(ttk.Frame):
    def __init__(self, master, presenter):
        self.master = master
        self.presenter = presenter

        self.fit_params = {}
        self.config_params = {}

        super().__init__(self.master)

        # Fit Parameters (uneditable)
        params_frame = ttk.Frame(self)
        params_frame.pack(side="left", expand=True, pady=15)
        keys = ["N", "A", "x0", "y0", "sx", "sy", "theta", "z0"]
        labels = ["N", "A", "x_0", "y_0", "σ_x", "σ_y", "θ", "z_0"]
        for l_idx, lbl in enumerate(labels):
            ttk.Label(params_frame, text=lbl).grid(row=l_idx, column=0)

        for f_idx in range(8):
            entry = ttk.Entry(params_frame, state="readonly")
            entry.grid(row=f_idx, column=1)
            self.fit_params[keys[f_idx]] = entry

        options_frame = ttk.Frame(self)
        options_frame.pack(side="left", expand=True, pady=15)

        roi_control = RegionOfInterestControl(options_frame)
        roi_control.pack(fill="x", expand=True)

        center_control = CenterControl(options_frame, self.presenter)
        center_control.pack(fill="x", expand=True)

        fit_frame = FitControl(options_frame)
        fit_frame.pack(fill="x", expand=True)

        rerun_fit_btn = ttk.Button(
            options_frame, text="Rerun Fit", command=self._rerun_fit
        )
        rerun_fit_btn.pack(fill="x", expand=True, padx=10, pady=5)

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

    def _rerun_fit(self):
        self.presenter.shot_presenter.refit_current_shot()


class RegionOfInterestControl(ttk.LabelFrame):
    """Creates ROI object for user-defined cropping and processing of cropped image in gaussian fitting."""
    def __init__(self, master):
        super().__init__(master, text="ROI")
        self.master = master
        ttk.Label(self, text="X").grid(row=0, column=1)
        ttk.Label(self, text="Y").grid(row=0, column=2)

        ttk.Label(self, text="Top Left").grid(row=1, column=0)
        ttk.Label(self, text="Bottom Right").grid(row=2, column=0)

        self.roi_entries = []  # x0, y0, x1, y1
        for row in range(1, 2 + 1):
            for col in range(1, 2 + 1):
                var = tk.IntVar(None)
                entry = FloatEntry(self, width=4, textvariable=var)
                entry.var = var
                entry.bind("<Return>", self._update_roi)
                entry.grid(row=row, column=col)
                self.roi_entries.append(entry)

                if config.roi:
                    var.set(config.roi[(row - 1) * 2 + (col - 1)])

        self.toggle_roi = ttk.Button(self, text="Enable", command=self._toggle_roi)
        self.toggle_roi.grid(row=3, column=1, columnspan=2)

    def _update_roi(self, event=None):
        try:
            roi = tuple(int(v.var.get()) for v in self.roi_entries)
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


class ThreeROI(ttk.Frame):
    """Creates object for three ROIs processing"""
    def __init__(self, master, presenter):
        self.master = master
        self.presenter = presenter

        super().__init__(self.master)

        options_frame = ttk.Frame(self)
        options_frame.pack(side="left", expand=True, pady=15)

        threeroi_control = ThreeRegionOfInterestControl(options_frame)
        threeroi_control.pack(fill="x", expand=True)

        rerun_fit_btn = ttk.Button(
            options_frame, text="Rerun Fit", command=self._rerun_fit
        )
        rerun_fit_btn.pack(fill="x", expand=True, padx=10, pady=5)

        counts_frame = ttk.Frame(self)
        counts_frame.pack(side="right", expand=True, pady=15)

        ttk.Label(counts_frame, text="Background subtracted atom count in ROI").grid(row=0, column=0, columnspan=2)

        labels = ["A", "B", "BG/px", "(A - B)/(A + B)"]
        keys = ["roia", "roib", "roibg", "a_b_ratio"]
        self.countentries = {}
        for l_indx, lbl in enumerate(labels):
            ttk.Label(counts_frame, text=lbl).grid(row=l_indx + 1, column=0)
            entry = ttk.Entry(counts_frame, state="readonly")
            entry.grid(row=l_indx + 1, column=1)
            self.countentries[keys[l_indx]] = entry

    def _rerun_fit(self):
        self.presenter.shot_presenter.refit_current_shot()

    def display(self, threeroi_counts):
        for k, v in threeroi_counts.items():
            entry = self.countentries[k]
            entry.configure(state="normal")
            entry.delete(0, "end")
            entry.insert(0, "{:.4g}".format(v))
            entry.configure(state="readonly")

class ThreeRegionOfInterestControl(ttk.LabelFrame):
    """Creates object for user input to define the three separate ROI for atom number counting."""
    def __init__(self, master):
        super().__init__(master, text="Three ROI")
        self.master = master

        style = ttk.Style()
        style.configure("R.TLabel", foreground="red")
        style.configure("B.TLabel", foreground="blue")
        style.configure("G.TLabel", foreground="green")

        # Defines labels for ROI grid
        # First ROI
        ttk.Label(self, text="ROI A", style="R.TLabel").grid(row=0, column=0)
        ttk.Label(self, text="X").grid(row=0, column=1)
        ttk.Label(self, text="Y").grid(row=0, column=2)
        ttk.Label(self, text="Top Left").grid(row=1, column=0)
        ttk.Label(self, text="Bottom Right").grid(row=2, column=0)
        # Second ROI
        ttk.Label(self, text="ROI B", style="B.TLabel").grid(row=3, column=0)
        ttk.Label(self, text="X").grid(row=3, column=1)
        ttk.Label(self, text="Y").grid(row=3, column=2)
        ttk.Label(self, text="Top Left").grid(row=4, column=0)
        ttk.Label(self, text="Bottom Right").grid(row=5,column=0)
        # Third ROI
        ttk.Label(self, text="ROI BG", style="G.TLabel").grid(row=6, column=0)
        ttk.Label(self, text="X").grid(row=6, column=1)
        ttk.Label(self, text="Y").grid(row=6, column=2)
        ttk.Label(self, text="Top Left").grid(row=7, column=0)
        ttk.Label(self,text="Bottom Right").grid(row=8, column=0)

        self.three_roi_entries = []

        for row in chain(range(1, 2 + 1), range(4, 5 + 1), range(7, 8 + 1)):
            for col in range(1, 2 + 1):
                var = tk.IntVar(None)
                entry = FloatEntry(self, width=12, textvariable=var)
                entry.var = var
                entry.bind("<Return>", self._update_three_roi)
                entry.grid(row=row, column=col)
                self.three_roi_entries.append(entry)

                if config.threeroi:
                    if row == 1 or row == 2:
                        var.set(config.threeroi[(row - 1) * 2 + (col - 1)])
                    elif row == 4 or row == 5:
                        var.set(config.threeroi[(row - 4) * 2 + (col - 1) + 4])
                    elif row == 7 or row == 8:
                        var.set(config.threeroi[(row - 7) * 2 + (col - 1) + 8])

        self.toggle_three_roi = ttk.Button(self, text="Enable", command=self._toggle_three_roi)
        self.toggle_three_roi.grid(row=9, column=1, columnspan=2)

    def _toggle_three_roi(self):
        if config.three_roi_enabled:
            config.three_roi_enabled = False
            self.toggle_three_roi.configure(text="Enable", state="normal")
            logging.debug("Three ROI disabled")
        else:
            if self._update_three_roi():
                config.three_roi_enabled = True
                self.toggle_three_roi.configure(text="Disable", state="active")
                logging.debug("Three ROI enabled")

    def _update_three_roi(self, event=None):
        try:
            threeroi = tuple(int(v.var.get()) for v in self.three_roi_entries)
        except ValueError:
            logging.error("Malformed Three ROI params!")
            return False
        
        if threeroi[0] < threeroi[2] and threeroi[1] < threeroi[3]:
            if threeroi[4] < threeroi[6] and threeroi[5] < threeroi[7]:
                if threeroi[8] < threeroi[10] and threeroi[9] < threeroi[11]:
                    config.threeroi = threeroi
                    config.save()
                    logging.info("Updated region of interest: %s", str(threeroi))
                    return True
                else:
                    logging.warning("Invalid ROI BG params!")
            else:
                logging.warning("Invalid ROI B params!")
        else:
            logging.warning("Invalid ROI A params!")

class CenterControl(ttk.LabelFrame):
    """Toggle object for fixing center of gaussian fit."""
    def __init__(self, master, presenter):
        self.master = master
        self.presenter = presenter

        super().__init__(self.master, text="Center")

        self.fixcenter = tk.BooleanVar()
        self.fixcenter.set(config.fix_center)
        fixcenter_btn = ttk.Checkbutton(
            self,
            text="Fix Center",
            variable=self.fixcenter,
            command=self._toggle_center,
        )
        fixcenter_btn.grid(row=0, column=0)

        center_x_var = tk.DoubleVar()
        self.center_x = FloatEntry(self, width=5, textvariable=center_x_var)
        self.center_x.var = center_x_var

        center_y_var = tk.DoubleVar()
        self.center_y = FloatEntry(self, width=5, textvariable=center_y_var)
        self.center_y.var = center_y_var

        if config.center:
            x, y = config.center
            self.center_x.var.set(x)
            self.center_y.var.set(y)

        self.center_x.grid(row=0, column=1)
        self.center_y.grid(row=0, column=2)

        self.center_x.bind("<Return>", self._update_center)
        self.center_y.bind("<Return>", self._update_center)

        self.current_shot_center = ttk.Button(
            self, text="Current Shot", command=self._fill_current_shot_center
        )
        self.current_shot_center.grid(row=1, column=1, columnspan=2)

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

    def _fill_current_shot_center(self):
        shot = self.presenter.shot_presenter.current_shot
        if shot:
            if shot.fit:
                self.center_x.var.set(shot.fit.best_values["x0"])
                self.center_y.var.set(shot.fit.best_values["y0"])
                self._update_center()
            else:
                logging.warning("Shot has no 2D Gaussian fit.")
        else:
            logging.error("No last shot to pull from!")


class FitControl(ttk.LabelFrame):
    """Toggle object for triggering fit and not fitting."""
    def __init__(self, master):
        super().__init__(master, text="Fit")
        self.master = master

        self.fix_z0 = tk.BooleanVar()
        self.fix_z0.set(config.fix_z0)
        fix_z0_btn = ttk.Checkbutton(
            self, text="Fix z0", variable=self.fix_z0, command=self._toggle_fix_z0
        )
        fix_z0_btn.grid(row=0, column=0, sticky="w")

        self.fix_theta = tk.BooleanVar()
        self.fix_theta.set(config.fix_theta)
        fix_theta_btn = ttk.Checkbutton(
            self, text="Fix θ", variable=self.fix_theta, command=self._toggle_fix_theta
        )
        fix_theta_btn.grid(row=0, column=1, sticky="w")

        self.enable_fit = tk.BooleanVar()
        self.enable_fit.set(config.fit)
        enable_fit_btn = ttk.Checkbutton(
            self,
            text="Enable Fitting",
            variable=self.enable_fit,
            command=self._toggle_fit,
        )
        enable_fit_btn.grid(row=1, column=0, sticky="w")

        self.enable_fluorescence = tk.BooleanVar()
        self.enable_fluorescence.set(config.fluor)
        enable_fluorescence_btn = ttk.Checkbutton(
            self,
            text="Enable Fluor",
            variable=self.enable_fluorescence,
            command=self._toggle_fluorescence
        )
        enable_fluorescence_btn.grid(row=1, column=1, sticky="w")

        self.fit_against = tk.StringVar(
            self, name="fit_against", value=self._get_fit_against()
        )
        fit_against_options = ("Optical Density", "Raw Absorption")
        ttk.Label(self, text="Fit Against: ").grid(row=2, column=0)
        fa_menu = ttk.OptionMenu(
            self,
            self.fit_against,
            self.fit_against.get(),
            *fit_against_options,
            command=self._set_fit_against,
        )
        fa_menu.configure(width=11)
        fa_menu.grid(row=2, column=1)

    def _get_fit_against(self):
        if config.fit_optical_density:
            return "Optical Density"
        else:
            return "Raw Absorption"

    def _set_fit_against(self, val):
        if val == "Optical Density":
            config.fit_optical_density = True
        else:
            config.fit_optical_density = False
        config.save()

    def _toggle_fix_theta(self):
        config.fix_theta = self.fix_theta.get()
        config.save()

    def _toggle_fit(self):
        config.fit = self.enable_fit.get()

    def _toggle_fix_z0(self):
        config.fix_z0 = self.fix_z0.get()
        config.save()

    def _toggle_fluorescence(self):
        config.fluor = self.enable_fluorescence.get()
        config.save()

class ExperimentParams(ttk.Frame):
    """Tab for viewing and reconfiguring some experimental parameters,
        "camera": {"pixel_size": "mm"},
        "beam": {
            "wavelength": "nm",
            "magnification": "x",
            "detuning": "MHz",
            "linewidth": "MHz",

    """
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.config_params = {}

        frame = ttk.Frame(self)
        frame.pack(expand=True)

        p_idx = 0
        for section in ("camera", "beam"):
            for key in config[section].keys():
                text = key.replace("_", " ").capitalize()
                ttk.Label(frame, text=text).grid(row=p_idx, column=0)

                units = type(config).units.get(section, {}).get(key)
                if units:
                    ttk.Label(frame, text=units).grid(row=p_idx, column=2)

                entry = FloatEntry(frame, state="normal")
                entry.grid(row=p_idx, column=1)
                entry.insert(0, config[section].getfloat(key))
                self.config_params[f"{section}.{key}"] = entry

                p_idx += 1

        save = ttk.Button(frame, text="Save", command=self._save_config)
        save.grid(row=p_idx, column=1)

    def _save_config(self):
        for name, entry in self.config_params.items():
            section, key = name.split(".")
            config[section][key] = str(entry.get())
        config.save()
