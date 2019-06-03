import signal
import time
import tkinter as tk

from tkinter import font
from tkinter import ttk

from gui.logs import LogTextBox, queue_handler
from gui.plots import MplFigure, FitParams, TemperatureParams, plot_queue


class Application(ttk.Frame):
    def __init__(self, master, threads=None):
        pad = 5

        super().__init__(master)
        self.threads = threads or []
        self.master = master
        self.master.title("Absorption Imager")
        self.pack(fill="both", expand=True)

        # Row 0 and column 1 expand when window is resized
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=2)
        self.grid_columnconfigure(1, weight=1)

        # Tabs
        tabs = ttk.Notebook(self)
        fp = FitParams(tabs)
        temp = TemperatureParams(tabs)
        tabs.add(fp, text="Gaussian")
        tabs.add(temp, text="Temperature")
        self.fit_params = fp
        tabs.grid(row=0, column=0)

        # Main image output
        image_frame = ttk.LabelFrame(self)
        image_frame.grid(row=0, column=1, rowspan=2, padx=pad, pady=pad, sticky="NSEW")
        self.figure = MplFigure(image_frame)

        # Logs
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


def start(threads):
    root = tk.Tk()
    root.state("zoomed")

    app = Application(root, threads)

    default_font = font.nametofont("TkTextFont")
    default_font.configure(size=18)

    menubar = tk.Menu(root)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=filemenu)

    root.config(menu=menubar)
    mainloop(app)
