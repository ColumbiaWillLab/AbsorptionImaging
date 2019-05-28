import re
import os
import sys
import time
import logging

from collections import defaultdict
from pathlib import Path
from watchdog.observers import Observer

import shots
import fitting
import density

from file_handling import create_handler, move_raw_images


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
    print(f"1: PROCESSING SHOT {name}")
    print("-------------------------------")
    shot = shots.Shot(paths)
    move_raw_images(paths)

    print("\n")
    print("2: GAUSSIAN FITTING (2D IMAGE)")
    print("-------------------------------")
    final_error, best, zoomed, int_error = fitting.two_D_gaussian(
        "automatic", 5, shot, 1
    )

    print("\n")
    print("3: GAUSSIAN FITTING (1D SLICES)")
    print("-------------------------------")
    fit_h, fit_v, param_h, param_v, x_hor, y_hor, x_ver, y_ver, x_axis, y_axis, horizontal, vertical = fitting.one_D_gaussian(
        shot, best
    )

    print("\n")
    print("4: PHYSICAL DENSITY ANALYSIS")
    print("-------------------------------")
    atom_num = density.atom_number(shot)
    print("Atom number: {:.2e}".format(atom_num))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.path.dirname(os.path.abspath(__file__))

    event_handler = create_handler(logging, _check_and_dispatch)

    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)
    observer.start()

    print("Watching for new files...")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
