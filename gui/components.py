import tkinter as tk
import tkinter.ttk as ttk


class FloatEntry(ttk.Entry):
    def __init__(self, master, widget=None, **kw):
        vcmd = (master.register(self.onValidate), "%P")
        kw["validate"] = "key"
        kw["validatecommand"] = vcmd
        super().__init__(master, widget, **kw)

    def onValidate(self, P):
        try:
            P == "" or float(P)
        except ValueError:
            self.bell()
            return False
        return True
