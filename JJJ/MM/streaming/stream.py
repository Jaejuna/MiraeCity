import threading
import sys, os
from queue import Queue

class loadThread(threading.Thread):
    def __init__(self, queue: Queue):
        threading.Thread.__init__(self)
        self.queue = queue
    pass

