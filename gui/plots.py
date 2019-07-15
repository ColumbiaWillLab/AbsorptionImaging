import tkinter.ttk as ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import config
from queues import event_queue, shot_list


class MplFigure(object):
    """Main frame for plots"""

    def __init__(self, master):
        self.master = master

        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # self.toolbar = NavigationToolbar2Tk(canvas, master)
        # self.toolbar.update()

    def display(self, obj, fit=True):
        obj.plot(self.figure, fit=fit)
        self.canvas.draw()


class ShotList(ttk.Treeview):
    def __init__(self, master, **kw):
        kw["columns"] = ("atoms", "sigma_x", "sigma_y")
        kw["selectmode"] = "browse"
        super().__init__(master, **kw)

        self.column("#0", anchor="w", width=200)
        self.column("atoms", anchor="w", width=100, stretch=False)
        self.column("sigma_x", anchor="w", width=100, stretch=False)
        self.column("sigma_y", anchor="w", width=100, stretch=False)

        self.heading("#0", text="Shot")
        self.heading("atoms", text="Atom Number")
        self.heading("sigma_x", text="Std. Dev. X")
        self.heading("sigma_y", text="Std. Dev. Y")

        self.pack(fill="both", expand=True)
        self.deque = shot_list

        self.bind("<Double-1>", self._on_double_click)
        self.bind("<Return>", self._on_return_keypress)
        # self.bind("<<TreeviewSelect>>", self._on_treeview_select)

    def add(self, shot):
        self.deque.append(shot)
        self.clear()
        for shot in self.deque:
            values = (
                shot.atom_number,
                shot.twoD_gaussian.best_values["sx"] * config.pixel_size,
                shot.twoD_gaussian.best_values["sy"] * config.pixel_size,
            )
            self.insert(
                "", "end", text=shot.name, values=tuple(map("{:.4g}".format, values))
            )

    def clear(self):
        self.delete(*self.get_children())

    def _on_double_click(self, event):
        idx = self.index(self.identify("item", event.x, event.y))
        event_queue.put((self.deque[idx], {"append": False}))

    def _on_return_keypress(self, event):
        idx = self.index(self.focus())
        event_queue.put((self.deque[idx], {"append": False}))

    def _on_treeview_select(self, event):
        idx = self.index(self.focus())
        event_queue.put((self.deque[idx], {"append": False}))
