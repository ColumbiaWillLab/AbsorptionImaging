from datetime import date
from pathlib import Path

from watchdog.events import PatternMatchingEventHandler


def create_handler(logging, callback):
    """Wrapper for the watchdog EventHandler to store file paths in a set"""
    bmp_paths = set()

    def on_created(e):
        bmp_paths.add(e.src_path)
        callback(bmp_paths)

    def on_deleted(e):
        bmp_paths.discard(e.src_path)

    event_handler = PatternMatchingEventHandler(
        patterns=["*.bmp"], ignore_directories=True, case_sensitive=False
    )
    event_handler.on_any_event = lambda e: logging.debug(e.src_path)
    event_handler.on_created = on_created
    event_handler.on_deleted = on_deleted

    return event_handler


def move_raw_images(paths, failed=False):
    """Move original images to "Raw Data" folder by date"""
    destination = Path("../Raw Data/").joinpath(str(date.today()))
    if failed:
        destination = destination.joinpath("failed")
    destination.mkdir(parents=True, exist_ok=True)

    for path in paths:
        path.replace(destination.joinpath(path.name))


def output_path(name):
    """Move processed images to "Analysis Results" folder by date"""
    output = Path("../Analysis Results/").joinpath(str(date.today()))
    output.mkdir(parents=True, exist_ok=True)
    return output.joinpath(f"{name}.png")
