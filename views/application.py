import platform

import tkinter as tk
import tkinter.font as font
import tkinter.ttk as ttk

from views.logs import LogTextBox
from views.settings import MplFigure, Settings
from views.shots import ShotList, ShotFit, ExperimentParams, ThreeROI
from views.sequences import ToFFit, AtomNumberOptimization


class MainWindow(ttk.Frame):
    """Main wrapper class for the GUI. Acts as parent frame for interior widgets."""

    def __init__(self, presenter):
        # Store references
        self.presenter = presenter

        # Initialize root Tk widget and state
        self.master = tk.Tk()
        self.master.title("Absorption Imager")
        self.master.state("zoomed")

        super().__init__(self.master)

        menubar = tk.Menu(self.master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        self.master.config(menu=menubar)

        if platform.system() == "Darwin":
            default_font = font.nametofont("TkTextFont")
            default_font.configure(size=18)
        else:
            default_font = font.nametofont("TkTextFont")
            default_font.configure(size=14)

        # Initialize our application window
        self.pack(fill="both", expand=True)

        # Row 0 and column 1 expand when window is resized
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=2)
        self.grid_columnconfigure(1, weight=1)

        # Draw interface
        self.tabs = Tabs(self, self.presenter)
        self.tabs.grid(row=0, column=0)

        pad = 5
        # Main image output
        plot_frame = ttk.LabelFrame(self)
        plot_frame.grid(row=0, column=1, rowspan=2, padx=pad, pady=pad, sticky="NSEW")
        self.plot = MplFigure(plot_frame)

        # Logs and TreeView
        left_bottom_frame = ttk.Frame(self)
        left_bottom_frame.grid(row=1, column=0, padx=pad, pady=pad, sticky="NSEW")

        shot_frame = ttk.Labelframe(left_bottom_frame, text="Shots")
        shot_frame.pack(fill="both", expand=False)
        self.shot_list = ShotList(shot_frame, self.presenter, height=8)

        log_frame = ttk.Labelframe(left_bottom_frame, text="Log", relief="sunken")
        log_frame.pack(fill="both", expand=True)
        self.logs = LogTextBox(log_frame)

    def mainloop(self):
        """There is some sort of Tk bug in OS X scroll handling. Wrap the Tk
        mainloop to recover if encountered."""
        try:
            super().mainloop()
        except UnicodeDecodeError:
            self.mainloop()


class Tabs(ttk.Notebook):
    def __init__(self, master, presenter):

        self.master = master
        self.presenter = presenter

        super().__init__(self.master)

        self.shot_fit = ShotFit(self, self.presenter)
        self.tof_fit = ToFFit(self, self.presenter)
        self.atom_number_fit = AtomNumberOptimization(self, self.presenter)
        self.three_roi_atom_count = ThreeROI(self, self.presenter)
        exp = ExperimentParams(self)
        self.settings = Settings(self)

        self.add(self.shot_fit, text="Gaussian", padding=10)
        self.add(self.tof_fit, text="Temperature", padding=10)
        self.add(self.atom_number_fit, text="Atom # Optimization", padding=10)
        self.add(self.three_roi_atom_count, text="Three ROIs", padding=10)
        self.add(exp, text="Experiment Settings", padding=10)
        self.add(self.settings, text="Settings", padding=10)
