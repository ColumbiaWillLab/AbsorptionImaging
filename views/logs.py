"""Tkinter ScrolledText widget to pull from logging.
Based off of https://github.com/beenje/tkinter-logging-text-widget"""
import tkinter.scrolledtext as tkst


class LogTextBox(tkst.ScrolledText):
    """Poll messages from a logging queue and display them in a scrolled text widget"""

    def __init__(self, master):
        self.master = master
        super().__init__(master, state="disabled", height=12)

        self.tag_config("INFO", foreground="black")
        self.tag_config("DEBUG", foreground="gray")
        self.tag_config("WARNING", foreground="orange")
        self.tag_config("ERROR", foreground="red")
        self.tag_config("CRITICAL", foreground="red", underline=1)
        self.pack(fill="both", expand=True)

    def display(self, msg, levelname):
        """Append log message to end of text block."""
        self.configure(state="normal")
        self.insert("end", msg + "\n", levelname)
        self.configure(state="disabled")
        self.see("end")

    def trim_log(self, num):
        """Trim log to MAX_LINES."""
        idx = float(self.index("end-1c"))
        if idx > num:
            self.configure(state="normal")
            self.delete("1.0", idx - num)
            self.configure(state="disabled")
