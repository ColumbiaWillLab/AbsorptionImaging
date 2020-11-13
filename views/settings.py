import logging

import tkinter as tk
import tkinter.ttk as ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import config


class MplFigure(ttk.Frame):
    """Main frame for plots"""

    def __init__(self, master):
        super().__init__(master)
        self.pack()

        self.figure = Figure(figsize=(8, 5))  # dummy figure for init
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # self.toolbar = NavigationToolbar2Tk(canvas, master)
        # self.toolbar.update()

    def display(self, obj):
        obj.plot(self.figure)
        self.canvas.draw()


class Settings(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        frame = ttk.Frame(self)
        frame.pack(expand=True)

        # Defines GUI for colormaps in plot
        self.colormap = tk.StringVar(
            frame, name="colormap", value=config.get("plot", "colormap")
        )
        color_options = ("cividis", "viridis", "Professor Mode")
        ttk.Label(frame, text="Colormap").grid(row=0, column=0)
        ttk.OptionMenu(
            frame,
            self.colormap,
            self.colormap.get(),
            *color_options,
            command=lambda val: self.save_config("colormap", val),
        ).grid(row=0, column=1)

        # Defines comment box for saving in logging.csv
        ttk.Label(frame, text="000_Logging.hdf5 comments").grid(row=2, column=0, columnspan=2)
        self.comment_string = ""
        self.var = tk.StringVar()
        comments_entry = ttk.Entry(frame, textvariable=self.var)
        comments_entry.grid(row=3, column=0, columnspan=3, padx=5, pady=5)
        #ttk.Label(frame, text="Any comments are automatically appended in 'logging.csv' .").grid(row=5, column=0, columnspan=3)

        # Updating comments strategy
        self.update_comments = ttk.Button(frame, text="Update", command=self._update_comments)
        self.update_comments.grid(row=4, column=1)
        comments_entry.bind('<Return>', self._update_comments) 

    def save_config(self, name, val):
        if val == "Professor Mode":
            config.set("plot", name, value="Greys")
        else:
            config.set("plot", name, value=val)
        config.save()

    def get_comment(self):
        return self.comment_string

    def _update_comments(self, event=None):
        self.comment_string = self.var.get()
        logging.info("Updated comments: %s", self.comment_string)

    # Include a comments section in GUI where the exported output goes to logging.csv.
