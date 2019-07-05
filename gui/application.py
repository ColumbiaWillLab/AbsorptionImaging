import signal
import tkinter as tk

from tkinter import font
from tkinter import ttk

from queues import event_queue
from models import shots, time_of_flight

from gui.logs import LogTextBox, queue_handler
from gui.plots import MplFigure, FitParams, TemperatureParams, PlotSettings, ShotList


class Application(ttk.Frame):
    """Main wrapper class for the GUI. Acts as parent frame for interior widgets.
    We also give it a reference to the other running threads to handle shutdown"""

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
        settings = PlotSettings(tabs)
        tabs.add(fp, text="Gaussian")
        tabs.add(temp, text="Temperature")
        tabs.add(settings, text="Plot Settings")
        tabs.grid(row=0, column=0)
        self.fit_params = fp
        self.temp_params = temp

        # Main image output
        image_frame = ttk.LabelFrame(self)
        image_frame.grid(row=0, column=1, rowspan=2, padx=pad, pady=pad, sticky="NSEW")
        self.figure = MplFigure(image_frame)

        # Logs and TreeView
        left_bottom_frame = ttk.Frame(self)
        left_bottom_frame.grid(row=1, column=0, padx=pad, pady=pad, sticky="NSEW")

        shot_frame = ttk.Labelframe(left_bottom_frame, text="Shots")
        shot_frame.pack(fill="both", expand=False)
        self.shot_list = ShotList(shot_frame, height=5)

        console_frame = ttk.Labelframe(left_bottom_frame, text="Log", relief="sunken")
        console_frame.pack(fill="both", expand=True)
        self.console = LogTextBox(console_frame, queue_handler.log_queue)

        # Handle window closure or SIGINT from console
        self.master.protocol("WM_DELETE_WINDOW", self.quit)
        signal.signal(signal.SIGINT, self.quit)

        self.after(100, self.poll_event_queue)

    def display_shot(self, shot, metadata):
        """Updates the data display with new info.
        The figure itself is updated by passing the figure reference around directly."""
        self.figure.display(shot, fit=metadata.get("fit", True))
        if metadata.get("fit", True):
            self.fit_params.display(
                dict({"N": shot.atom_number}, **shot.twoD_gaussian.best_values)
            )
        else:
            self.fit_params.clear()
        if metadata.get("append", True):
            self.shot_list.add(shot)

    def display_tof(self, tof, metadata):
        self.figure.display(tof)
        self.temp_params.display(tof)

    def poll_event_queue(self):
        """The plot queue is polled every 100ms for updates."""
        if not event_queue.empty():
            obj, metadata = event_queue.get(block=False)
            if isinstance(obj, shots.Shot):
                self.display_shot(obj, metadata)
            elif isinstance(obj, time_of_flight.TimeOfFlight):
                self.display_tof(obj, metadata)
        self.after(100, self.poll_event_queue)

    def quit(self, *args):
        """Shut down Tkinter master and stop/join all threads."""
        for thread in self.threads:
            thread.stop()

        for thread in self.threads:
            thread.join(3)

        self.master.destroy()
        self.master.quit()


def mainloop(app):
    """There is some sort of Tk bug in OS X scroll handling. Wrap the Tk
    mainloop to recover if encountered."""
    try:
        app.mainloop()
    except UnicodeDecodeError:  # Tk bug in OS X scroll handling
        mainloop(app)


def start(threads):
    """Construct the root Tk window and start the GUI mainloop."""
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
