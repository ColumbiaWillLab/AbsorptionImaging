import queue

from collections import deque

event_queue = queue.Queue()

shot_list = deque(maxlen=5)
