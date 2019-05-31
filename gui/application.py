import signal
import time
import tkinter as tk

from tkinter import ttk

from gui.logging import LogTextBox, queue_handler
from gui.plotting import MplFigure, figure_queue


class Application(ttk.Frame):
    def __init__(self, master, threads=[]):
        super().__init__(master)
        self.threads = threads
        self.master = master
        self.master.title("Absorption Imager")
        self.pack(fill=tk.BOTH, expand=True)

        image_frame = ttk.LabelFrame(self, text="Image")
        image_frame.grid(row=1, column=2)
        self.figure = MplFigure(image_frame, figure_queue)

        console_frame = ttk.Labelframe(self, text="Console")
        console_frame.grid(row=2, column=2)
        self.console = LogTextBox(console_frame, queue_handler.log_queue)

        self.master.protocol("WM_DELETE_WINDOW", self.quit)
        signal.signal(signal.SIGINT, self.quit)

    def quit(self, *args):
        """Shut down Tkinter master and stop/join all threads"""
        for thread in self.threads:
            thread.stop()

        for thread in self.threads:
            thread.join(3)

        self.master.quit()
        self.master.destroy()


def start(threads=[]):
    root = tk.Tk()
    root.state("zoomed")

    app = Application(root, threads)
    app.mainloop()
