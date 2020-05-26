import signal
import queue

from models import Shot

from .logs import LogPresenter
from .shots import ShotPresenter
from .sequences import SequencePresenter


class MainWindowPresenter:
    def __init__(self, worker, *, shutdown_cleanup, log_queue):
        self.worker = worker
        self.event_queue = queue.Queue()
        self.log_queue = log_queue
        self.shutdown_cleanup = shutdown_cleanup

        # Placeholders for view dependent initialization
        # TODO: is there a way to eliminate the circular dependency?
        self.view = None
        self.log_presenter = None
        self.shot_presenter = None
        self.sequence_presenter = None

    def set_view(self, view):
        self.view = view
        self.log_presenter = LogPresenter(self.view.logs, log_queue=self.log_queue)
        self.shot_presenter = ShotPresenter(
            self,
            self.worker,
            plot_view=self.view.plot,
            fit_view=self.view.tabs.shot_fit,
            list_view=self.view.shot_list,
            threeroi_view=self.view.tabs.three_roi_atom_count,
            settings_view=self.view.tabs.settings
        )
        self.sequence_presenter = SequencePresenter(
            self, self.worker, plot_view=self.view.plot, fit_view=self.view.tabs.tof_fit
        )

        # Handle window closure or SIGINT from console
        self.shutdown_cleanup = self.shutdown_cleanup
        self.view.master.protocol("WM_DELETE_WINDOW", self.quit)
        signal.signal(signal.SIGINT, self.quit)

        # Start polling event queue
        self.view.after(100, self._poll_event_queue)

    def quit(self, *args, **kwargs):
        """Run callback, then shut down Tkinter master."""
        self.shutdown_cleanup()

        self.view.master.destroy()
        self.view.master.quit()

    def queue(self, func, *args, **kwargs):
        """Add object to event queue (only way to communicate between threads)."""
        return self.event_queue.put((func, args, kwargs))

    def _poll_event_queue(self):
        """The plot queue is polled every 100ms for updates."""
        if not self.event_queue.empty():
            obj = self.event_queue.get(block=False)
            if isinstance(obj, tuple):
                if len(obj) == 1:
                    obj[0]()
                elif len(obj) == 2:
                    if isinstance(obj[1], list):
                        obj[0](*obj[1])
                    elif isinstance(obj[1], dict):
                        obj[0](**obj[1])
                elif len(obj) == 3:
                    obj[0](*obj[1], **obj[2])
        self.view.after(100, self._poll_event_queue)
