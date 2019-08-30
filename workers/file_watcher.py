import re
import logging
import traceback

from collections import defaultdict
from pathlib import Path
from datetime import date


from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


class FileWatcher(Observer):
    """Wrap watchdog.observers.Observer to handle Shot BMP observation."""

    def __init__(self, directory, *, process_shot):
        super().__init__()

        self.process_shot = process_shot

        self.event_handler = _create_handler(self._check_and_dispatch)
        self.schedule(self.event_handler, directory, recursive=False)

    def start(self):
        super().start()
        logging.info("Watching for new files...")

    def _check_and_dispatch(self, bmp_paths):
        shot_bmps = defaultdict(int)
        all_paths = {}

        for path in map(Path, bmp_paths):
            if path.is_file():  # check for existence
                match = re.match(r"^(.*)(?:-|_)(\d)$", path.stem)
                if match:
                    name, idx = match.groups()
                    shot_bmps[name] += int(idx)
                    all_paths[f"{name}-{idx}"] = path

        for name, num in shot_bmps.items():
            if num == 6:  # Crude determination that all 3 images exist: 1 + 2 + 3
                paths = [all_paths[f"{name}-{num}"] for num in range(1, 4)]
                try:
                    self.process_shot(name, paths)
                    _move_raw_images(paths, failed=False)
                except Exception as e:
                    logging.error(traceback.format_exc())
                    _move_raw_images(paths, failed=True)


def _create_handler(callback):
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


def _move_raw_images(paths, failed=False):
    """Move original images to "Raw Data" folder by date"""
    destination = Path("../Raw Data/").joinpath(str(date.today()))
    if failed:
        destination = destination.joinpath("failed")
    destination.mkdir(parents=True, exist_ok=True)

    for path in paths:
        path.replace(destination.joinpath(path.name))
