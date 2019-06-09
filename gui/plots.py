import queue

from math import isfinite

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as tkst

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from config import config

from .components import FloatEntry

figure = Figure(figsize=(8, 5))
plot_queue = queue.Queue()


class MplFigure(object):
    """Main frame for plots"""

    def __init__(self, master):
        self.master = master

        canvas = FigureCanvasTkAgg(figure, master=master)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

        # toolbar = NavigationToolbar2Tk(canvas, master)
        # toolbar.update()

        self.canvas = canvas
        # self.toolbar = toolbar

    def display(self):
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

        save = ttk.Button(self, text="Save", command=self.save_config)
        save.grid(row=p_idx, column=1)

        labels = ["N", "A", "x_0", "y_0", "σ_x", "σ_y", "θ", "z_0"]
        for l_idx, lbl in enumerate(labels):
            ttk.Label(self, text=lbl).grid(row=l_idx, column=2)

        for f_idx in range(8):
            entry = ttk.Entry(self, state="readonly")
            entry.grid(row=f_idx, column=3)
            self.fit_params.append(entry)

    def display(self, fit_params):
        keys = ["N", "A", "x0", "y0", "sx", "sy", "theta", "z0"]
        for i, k in enumerate(keys):
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

    def save_config(self):
        for name, entry in self.config_params.items():
            section, key = name.split(".")
            config[section][key] = str(entry.get())

        config.save()


class TemperatureParams(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        ttk.Label(self, text="Times of Flight:").pack()

        st = tkst.ScrolledText(self, state="normal", height=12, width=10)
        st.pack()

        btn = ttk.Button(self, text="Run", command=self.run)
        btn.pack()

        self.st = st
        self.btn = btn

    def run(self):
        try:
            vals = [float(x) for x in self.st.get("1.0", "end-1c").split()]
            if not all(x >= 0 and isfinite(x) for x in vals):
                raise ValueError
        except ValueError:
            tk.messagebox.showerror("Temperature Fitting Alert", "Invalid Entry!")
