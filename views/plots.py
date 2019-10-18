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


class PlotSettings(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        frame = ttk.Frame(self)
        frame.pack(expand=True)

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

    def save_config(self, name, val):
        if val == "Professor Mode":
            config.set("plot", name, value="Greys")
        else:
            config.set("plot", name, value=val)
        config.save()
