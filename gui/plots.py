import tkinter as tk
import tkinter.ttk as ttk
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
import queue

from matplotlib.figure import Figure

figure = Figure(figsize=(8, 5), dpi=100)
plot_queue = queue.Queue()


class MplFigure(object):
    def __init__(self, master):
        self.master = master

        canvas = FigureCanvasTkAgg(figure, master=master)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

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

        for lbl in ["A", "x_0", "y_0", "σ_x", "σ_y", "θ", "N"]:
            ttk.Label(self, text=lbl).grid(column=0)

        self.entries = []
        for i in range(7):
            entry = ttk.Entry(self, state="readonly")
            entry.grid(row=i, column=1)
            self.entries.append(entry)

    def display(self, fit_params):
        for i, p in enumerate(fit_params[0:7]):
            entry = self.entries[i]
            entry.configure(state="normal")
            entry.delete(0, tk.END)
            entry.insert(0, p)
            entry.configure(state="readonly")
