import re
import logging
import traceback

from collections import defaultdict
from pathlib import Path
from watchdog.observers import Observer
from matplotlib.figure import Figure

from config import config
from queues import event_queue
from models import shots, time_of_flight
from .utils import create_handler, move_raw_images, output_path

tof = time_of_flight.TimeOfFlight([])


def start_tof(times):
    global tof
    tof = time_of_flight.TimeOfFlight(times)
    logging.info("Starting Time of Flight: %s", str(times))


def _check_and_dispatch(bmp_paths):
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
        if num == 6:  # 1 + 2 + 3
            paths = [all_paths[f"{name}-{num}"] for num in range(1, 4)]
            try:
                _process_shot(name, paths)
                move_raw_images(paths)
            except Exception as e:
                logging.error(traceback.format_exc())
                move_raw_images(paths, failed=True)


def _process_shot(name, paths):
    global tof

    logging.info("1: PROCESSING SHOT %s", name)
    logging.info("-------------------------------")

    shot = shots.Shot(name, paths)
    shot.warm_cache(config.fit)
    event_queue.put((shot, {"fit": False, "append": False}))
    if config.fit:
        event_queue.put((shot, {}))  # Call twice - display transmission before fitting

    savefig = Figure(figsize=(8, 5))
    shot.plot(savefig, fit=config.fit)
    savefig.savefig(output_path(name), dpi=150)

    tof.add(shot, lambda x: event_queue.put((x, {})))


def start_observer(directory):
    event_handler = create_handler(logging, _check_and_dispatch)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    logging.info("Watching for new files...")
    return observer
