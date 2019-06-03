import queue

import tkinter as tk
import tkinter.ttk as ttk
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from config import config

from .components import FloatEntry

figure = Figure(figsize=(8, 5), dpi=100)
plot_queue = queue.Queue()


class MplFigure(object):
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


class FitParams(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        p_idx = 0
        for section in config.sections():
            for key in config[section].keys():
                text = key.replace("_", " ").capitalize()
                ttk.Label(self, text=text).grid(row=p_idx, column=0)
                entry = FloatEntry(self, state="normal")
                entry.grid(row=p_idx, column=1)
                entry.insert(0, config[section].getfloat(key))
                p_idx += 1

        save = ttk.Button(self, text="Save")
        save.grid(row=p_idx, column=1)

        labels = ["A", "x_0", "y_0", "σ_x", "σ_y", "θ", "z_0"]
        for l_idx, lbl in enumerate(labels):
            ttk.Label(self, text=lbl).grid(row=l_idx, column=2)

        self.entries = []
        for f_idx in range(7):
            entry = ttk.Entry(self, state="readonly")
            entry.grid(row=f_idx, column=3)
            self.entries.append(entry)

    def display(self, fit_params):
        for i, p in enumerate(fit_params[0:7]):
            entry = self.entries[i]
            entry.configure(state="normal")
            entry.delete(0, "end")
            entry.insert(0, "{:.4g}".format(p))
            entry.configure(state="readonly")


class TemperatureParams(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
