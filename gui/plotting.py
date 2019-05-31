import tkinter as tk
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
import queue

from matplotlib.figure import Figure


class MplFigure(object):
    def __init__(self, master, queue=None):
        self.master = master
        self.queue = queue

        canvas = FigureCanvasTkAgg(figure, master=master)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # toolbar = NavigationToolbar2Tk(canvas, master)
        # toolbar.update()

        self.canvas = canvas
        # self.toolbar = toolbar

        self.master.after(100, self.poll_queue)

    def display(self, fig):
        self.canvas.draw()

    def poll_queue(self):
        while not self.queue.empty():
            fig = self.queue.get(block=False)
            self.display(fig)
        self.master.after(100, self.poll_queue)


figure = Figure(figsize=(5, 4), dpi=100)
figure_queue = queue.Queue()
