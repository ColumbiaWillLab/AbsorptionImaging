from math import isfinite
from collections import deque

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as tkst

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from config import config
from queues import shot_queue

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

    def display(self, shot):
        shot.plot(self.figure)
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


class ShotList(ttk.Treeview):
    def __init__(self, master, **kw):
        kw["columns"] = ("atoms", "sigma_x", "sigma_y")
        kw["selectmode"] = "browse"
        super().__init__(master, **kw)

        self.column("#0", anchor="w", width=200)
        self.column("atoms", anchor="w", width=100, stretch=False)
        self.column("sigma_x", anchor="w", width=100, stretch=False)
        self.column("sigma_y", anchor="w", width=100, stretch=False)

        self.heading("#0", text="Shot", anchor="w")
        self.heading("atoms", text="Atom Number", anchor="w")
        self.heading("sigma_x", text="Std. Dev. X", anchor="w")
        self.heading("sigma_y", text="Std. Dev. Y", anchor="w")

        self.pack(fill="both", expand=True)
        self.deque = deque(maxlen=5)

        self.bind("<Double-1>", self.on_double_click)
        self.bind("<Return>", self.on_return_keypress)

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

    def on_double_click(self, event):
        item = self.item(self.identify("item", event.x, event.y), "text")
        for shot in self.deque:
            if shot.name == item:
                shot_queue.put((shot, {"new": False}))
                break

    def on_return_keypress(self, event):
        item = self.item(self.focus(), "text")
        for shot in self.deque:
            if shot.name == item:
                shot_queue.put((shot, {"new": False}))
                break
