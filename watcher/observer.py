import re
import logging
import traceback

from collections import defaultdict
from pathlib import Path
from watchdog.observers import Observer
from matplotlib.figure import Figure

from models import shots
from gui.plots import figure, plot_queue
from .utils import create_handler, move_raw_images, output_path


def _check_and_dispatch(bmp_paths):
    shot_bmps = defaultdict(int)
    paths = {}

    for path in map(Path, bmp_paths):
        if path.is_file():  # check for existence
            match = re.match(r"^(.*)(?:-|_)(\d)$", path.stem)
            if match:
                name, idx = match.groups()
                shot_bmps[name] += int(idx)
                paths[f"{name}-{idx}"] = path

    for name, num in shot_bmps.items():
        if num == 6:  # 1 + 2 + 3
            try:
                _process_shot(name, [paths[f"{name}-{num}"] for num in range(1, 4)])
            except Exception as e:
                logging.error(traceback.format_exc())


def _process_shot(name, paths):
    logging.info("1: PROCESSING SHOT %s", name)
    logging.info("-------------------------------")
    shot = shots.Shot(name, paths)

    logging.info("\n")
    logging.info("2: GAUSSIAN FITTING")
    logging.info("-------------------------------")
    shot.twoD_gaussian
    shot.oneD_gaussians

    shot.plot(figure)
    plot_queue.put(
        (dict({"N": shot.atom_number}, **shot.twoD_gaussian.best_values), True)
    )

    savefig = Figure(figsize=(8, 5))
    shot.plot(savefig)
    savefig.savefig(output_path(name), dpi=150)

    move_raw_images(paths)


def start_observer(directory):
    event_handler = create_handler(logging, _check_and_dispatch)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    logging.info("Watching for new files...")
    return observer
