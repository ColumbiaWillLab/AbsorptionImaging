"""
General threading helpers
"""
import functools
import threading


def mainthread(func):
    """Assert that this function is being called on the main thread."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert threading.current_thread() is threading.main_thread()
        return func(*args, **kwargs)

    return wrapper
