import logging
import sys
import os

from watcher.observer import start_observer
from gui.application import start
from gui.logs import queue_handler


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[queue_handler],
    )

    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.path.dirname(os.path.abspath(__file__))

    observer = start_observer(directory)

    start([observer])


if __name__ == "__main__":
    main()
