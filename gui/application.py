import signal
import time
import tkinter as tk

from tkinter import ttk

from gui.logs import LogTextBox, queue_handler
from gui.plots import MplFigure, FitParams, plot_queue


class Application(ttk.Frame):
    def __init__(self, master, threads=[]):
        pad = 5

        super().__init__(master)
        self.threads = threads
        self.master = master
        self.master.title("Absorption Imager")
        self.pack(fill="both", expand=True)

        self.fit_params = FitParams(self)
        self.fit_params.grid(row=0, column=0)

        image_frame = ttk.LabelFrame(self)
        image_frame.grid(row=0, column=1, rowspan=2, padx=pad, pady=pad, sticky="NSEW")
        self.figure = MplFigure(image_frame)

        console_frame = ttk.Labelframe(self, text="Log", relief="sunken")
        console_frame.grid(row=1, column=0, padx=pad, pady=pad, sticky="NSEW")
        self.console = LogTextBox(console_frame, queue_handler.log_queue)

        self.master.protocol("WM_DELETE_WINDOW", self.quit)
        signal.signal(signal.SIGINT, self.quit)

        self.after(100, self.poll_plot_queue)

    def display(self, plot):
        self.fit_params.display(plot[0])
        self.figure.display()

    def poll_plot_queue(self):
        while not plot_queue.empty():
            plot = plot_queue.get(block=False)
            self.display(plot)
        self.after(100, self.poll_plot_queue)

    def quit(self, *args):
        """Shut down Tkinter master and stop/join all threads"""
        for thread in self.threads:
            thread.stop()

        for thread in self.threads:
            thread.join(3)

        self.master.quit()
        self.master.destroy()


def mainloop(app):
    try:
        app.mainloop()
    except UnicodeDecodeError:
        mainloop(app)


def start(threads=[]):
    root = tk.Tk()
    root.state("zoomed")

    app = Application(root, threads)
    mainloop(app)
