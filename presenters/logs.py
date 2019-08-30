class LogPresenter:
    MAX_LINES = 1000

    def __init__(self, view, *, log_queue):
        self.view = view
        self.log_queue = log_queue

        self.view.after(100, self.poll_log_queue)

    def poll_log_queue(self):
        """Poll and display all new messages in log_queue."""
        while not self.log_queue.empty():
            record = self.log_queue.get(block=False)
            self.view.display(*record)
            self.view.trim_log(type(self).MAX_LINES)
        self.view.after(100, self.poll_log_queue)
