"""Tkinter ScrolledText widget to pull from logging.
Based off of https://github.com/beenje/tkinter-logging-text-widget"""
import logging
import queue

import tkinter as tk
import tkinter.scrolledtext as tkst


class QueueHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_queue = queue.Queue()

    def emit(self, record):
        self.log_queue.put((self.format(record), record.levelname))


class LogTextBox(object):
    """Poll messages from a logging queue and display them in a scrolled text widget"""

    MAX_LINES = 1000

    def __init__(self, master, log_queue):
        self.master = master
        self.log_queue = log_queue

        st = tkst.ScrolledText(master, state="disabled", height=12)
        st.tag_config("INFO", foreground="black")
        st.tag_config("DEBUG", foreground="gray")
        st.tag_config("WARNING", foreground="orange")
        st.tag_config("ERROR", foreground="red")
        st.tag_config("CRITICAL", foreground="red", underline=1)
        st.pack(fill="both", expand=True)

        self.st = st
        master.after(100, self.poll_log_queue)

    def display(self, msg, levelname):
        st = self.st
        st.configure(state="normal")
        st.insert("end", msg + "\n", levelname)
        st.configure(state="disabled")
        st.see("end")

    def trim_log(self):
        st = self.st
        idx = float(self.st.index("end-1c"))
        max_lines = type(self).MAX_LINES
        if idx > max_lines:
            st.configure(state="normal")
            st.delete("1.0", idx - max_lines)
            st.configure(state="disabled")

    def poll_log_queue(self):
        while not self.log_queue.empty():
            record = self.log_queue.get(block=False)
            self.display(*record)
            self.trim_log()
        self.master.after(100, self.poll_log_queue)


queue_handler = QueueHandler()
