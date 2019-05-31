import re
import logging

from collections import defaultdict
from pathlib import Path
from watchdog.observers import Observer

import shots
import fitting
import density
import plotting

from .utils import create_handler, move_raw_images

from gui.plotting import figure, figure_queue


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
            _process_shot(name, [paths[f"{name}-{num}"] for num in range(1, 4)])


def _process_shot(name, paths):
    logging.info("1: PROCESSING SHOT %s", name)
    logging.info("-------------------------------")
    shot = shots.Shot(paths)

    logging.info("\n")
    logging.info("2: GAUSSIAN FITTING (2D IMAGE)")
    logging.info("-------------------------------")
    final_error, best, zoomed, int_error = fitting.two_D_gaussian(
        "automatic", 5, shot, 1
    )

    logging.info("\n")
    logging.info("3: GAUSSIAN FITTING (1D SLICES)")
    logging.info("-------------------------------")
    fit_h, fit_v, param_h, param_v, x_hor, y_hor, x_ver, y_ver, x_axis, y_axis, horizontal, vertical = fitting.one_D_gaussian(
        shot, best
    )

    logging.info("\n")
    logging.info("4: PHYSICAL DENSITY ANALYSIS")
    logging.info("-------------------------------")
    atom_num = density.atom_number(shot)
    logging.info("Atom number: %.2e", atom_num)

    plotting.plot(figure, shot)
    figure_queue.put(1)

    move_raw_images(paths)


def start_observer(directory):
    event_handler = create_handler(logging, _check_and_dispatch)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()
    logging.info("Watching for new files...")
    return observer
