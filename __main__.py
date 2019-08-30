"""
Main entry point of application (executable as folder or zip)
"""

import logging
import sys
import os
import queue

from concurrent.futures import ThreadPoolExecutor

from views.application import MainWindow
from presenters import MainWindowPresenter
from workers.file_watcher import FileWatcher


class App:
    """Main application - starts GUI, watcher, workers, etc."""

    def __init__(self, *, watch_directory):
        # Initialize logging
        self.log_handler = QueueHandler()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[self.log_handler],
        )

        # Initialize component instances
        self.worker = ThreadPoolExecutor(max_workers=1)

        self.app = MainWindowPresenter(
            self.worker,
            shutdown_cleanup=self.cleanup,
            log_queue=self.log_handler.log_queue,
        )
        self.gui = MainWindow(self.app)
        self.app.set_view(self.gui)

        self.file_watcher = FileWatcher(
            watch_directory, process_shot=self.app.shot_presenter.process_shot
        )

    def start(self):
        """Run file watcher and start interface."""
        self.file_watcher.start()
        self.gui.mainloop()

    def cleanup(self):
        """Pre-GUI shutdown cleanup tasks."""
        self.worker.shutdown(wait=False)

        self.file_watcher.stop()
        self.file_watcher.join(3)


class QueueHandler(logging.Handler):
    """Log handler that outputs to a queue."""

    def __init__(self):
        super().__init__()
        self.log_queue = queue.Queue()

    def emit(self, record):
        self.log_queue.put((self.format(record), record.levelname))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.path.dirname(os.path.abspath(__file__))

    app = App(watch_directory=directory)
    app.start()
