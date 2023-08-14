import threading 
import sys, os

class preprocessThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    pass



class inferenceThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    pass