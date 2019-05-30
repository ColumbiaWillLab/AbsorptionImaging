import signal
import tkinter as tk

from gui.logging import LogTextBox, queue_handler


class Application(tk.Frame):
    def __init__(self, master, threads=[]):
        super().__init__(master)
        self.threads = threads
        self.master = master
        self.master.title("Absorption Imager")
        self.pack(fill=tk.BOTH, expand=True)
        LogTextBox(self, queue_handler.log_queue)

        self.master.protocol("WM_DELETE_WINDOW", self.quit)
        signal.signal(signal.SIGINT, self.quit)

    def quit(self, *args):
        self.master.destroy()
        for thread in self.threads:
            thread.stop()
            thread.join()


def start(threads=[]):
    root = tk.Tk()
    root.geometry("1024x768")

    app = Application(root, threads)
    app.mainloop()
