from math import isfinite
from collections import deque

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as tkst

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from config import config
from queues import event_queue
from watcher import observer

from .components import FloatEntry


class MplFigure(object):
    """Main frame for plots"""

    def __init__(self, master):
        self.master = master

        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # self.toolbar = NavigationToolbar2Tk(canvas, master)
        # self.toolbar.update()

    def display(self, obj, fit=True):
        obj.plot(self.figure, fit=fit)
        self.canvas.draw()


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


class FitParams(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.fit_params = []
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

        self.fitvar = tk.BooleanVar()
        self.fitvar.set(config.fit)
        fitbtn = ttk.Checkbutton(
            self, text="Enable Fitting", variable=self.fitvar, command=self._toggle_fit
        )
        fitbtn.grid(row=p_idx + 1, column=1)

        labels = ["N", "A", "x_0", "y_0", "σ_x", "σ_y", "θ", "z_0"]
        for l_idx, lbl in enumerate(labels):
            ttk.Label(self, text=lbl).grid(row=l_idx, column=2)

        for f_idx in range(8):
            entry = ttk.Entry(self, state="readonly")
            entry.grid(row=f_idx, column=3)
            self.fit_params.append(entry)

    @property
    def keys(self):
        return ["N", "A", "x0", "y0", "sx", "sy", "theta", "z0"]

    def display(self, fit_params):
        for i, k in enumerate(self.keys):
            p = fit_params[k]
            if k in ["x0", "y0", "sx", "sy"]:
                p *= config.pixel_size
            elif k == "theta":
                p = np.degrees(p)
            entry = self.fit_params[i]
            entry.configure(state="normal")
            entry.delete(0, "end")
            entry.insert(0, "{:.4g}".format(p))
            entry.configure(state="readonly")

    def clear(self):
        for i, k in enumerate(self.keys):
            entry = self.fit_params[i]
            entry.configure(state="normal")
            entry.delete(0, "end")
            entry.configure(state="readonly")

    def _save_config(self):
        for name, entry in self.config_params.items():
            section, key = name.split(".")
            config[section][key] = str(entry.get())

        config.save()

    def _toggle_fit(self):
        config.fit = self.fitvar.get()


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
        st.insert("1.0", "0.25\n0.5\n0.75\n1\n1.25\n1.5\n1.75\n2\n2.25\n2.5")

        btn = ttk.Button(left_frame, text="Run", command=self.run)
        btn.pack()

        self.st = st
        self.btn = btn

        # Right Top Frame (Temperature Output)
        rt_frame = ttk.Frame(self)
        rt_frame.grid(row=0, column=1)

        ttk.Label(rt_frame, text="X Temp.").grid(row=0, column=0)
        ttk.Label(rt_frame, text="Y Temp.").grid(row=1, column=0)

        x_temp_entry = ttk.Entry(rt_frame, state="readonly", width=10)
        y_temp_entry = ttk.Entry(rt_frame, state="readonly", width=10)

        x_temp_entry.grid(row=0, column=1)
        y_temp_entry.grid(row=1, column=1)

        ttk.Label(rt_frame, text="Std. Error").grid(row=0, column=2)
        ttk.Label(rt_frame, text="Std. Error").grid(row=1, column=2)

        x_temp_err = ttk.Entry(rt_frame, state="readonly", width=10)
        y_temp_err = ttk.Entry(rt_frame, state="readonly", width=10)

        x_temp_err.grid(row=0, column=3)
        y_temp_err.grid(row=1, column=3)

        ttk.Label(rt_frame, text="Mean Atom #").grid(row=2, column=0)
        atom_n_mean = ttk.Entry(rt_frame, state="readonly", width=10)
        atom_n_mean.grid(row=2, column=1)

        ttk.Label(rt_frame, text="Coeff. Var.").grid(row=2, column=2)
        atom_n_cv = ttk.Entry(rt_frame, state="readonly", width=10)
        atom_n_cv.grid(row=2, column=3)

        self.x_temp_entry = x_temp_entry
        self.y_temp_entry = y_temp_entry
        self.x_temp_err = x_temp_err
        self.y_temp_err = y_temp_err
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
        except ValueError:
            tk.messagebox.showerror("Temperature Fitting Alert", "Invalid Entry!")
        observer.start_tof(vals)

    def display(self, tof):
        atom_n_mean = np.mean(tof.atom_number)
        pairs = [
            (self.x_temp_entry, tof.x_temp),
            (self.y_temp_entry, tof.y_temp),
            (self.x_temp_err, tof.x_temp_err),
            (self.y_temp_err, tof.y_temp_err),
            (self.atom_n_mean, atom_n_mean),
            (self.atom_n_cv, np.std(tof.atom_number) / atom_n_mean * 100),
        ]
        for entry, value in pairs:
            entry.configure(state="normal")
            entry.delete(0, "end")
            entry.insert(0, "{:.4g}".format(value))
            entry.configure(state="readonly")


class ShotList(ttk.Treeview):
    def __init__(self, master, **kw):
        kw["columns"] = ("atoms", "sigma_x", "sigma_y")
        kw["selectmode"] = "browse"
        super().__init__(master, **kw)

        self.column("#0", anchor="w", width=200)
        self.column("atoms", anchor="w", width=100, stretch=False)
        self.column("sigma_x", anchor="w", width=100, stretch=False)
        self.column("sigma_y", anchor="w", width=100, stretch=False)

        self.heading("#0", text="Shot")
        self.heading("atoms", text="Atom Number")
        self.heading("sigma_x", text="Std. Dev. X")
        self.heading("sigma_y", text="Std. Dev. Y")

        self.pack(fill="both", expand=True)
        self.deque = deque(maxlen=5)

        self.bind("<Double-1>", self._on_double_click)
        self.bind("<Return>", self._on_return_keypress)
        # self.bind("<<TreeviewSelect>>", self._on_treeview_select)

    def add(self, shot):
        self.deque.append(shot)
        self.clear()
        for shot in self.deque:
            values = (
                shot.atom_number,
                shot.twoD_gaussian.best_values["sx"] * config.pixel_size,
                shot.twoD_gaussian.best_values["sy"] * config.pixel_size,
            )
            self.insert(
                "",
                "end",
                text=shot.name,
                values=tuple(map(lambda x: "{:.4g}".format(x), values)),
            )

    def clear(self):
        self.delete(*self.get_children())

    def _on_double_click(self, event):
        idx = self.index(self.identify("item", event.x, event.y))
        event_queue.put((self.deque[idx], {"append": False}))

    def _on_return_keypress(self, event):
        idx = self.index(self.focus())
        event_queue.put((self.deque[idx], {"append": False}))

    def _on_treeview_select(self, event):
        idx = self.index(self.focus())
        event_queue.put((self.deque[idx], {"append": False}))
