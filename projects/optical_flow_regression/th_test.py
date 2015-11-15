__author__ = 'jiri'

import threading
import time
import math
import numpy as np
from multiprocessing import Pool


class myThread (threading.Thread):

    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        while True:
            start = time.time()
            for i in range(1000000):
                x = 1256456
                x = math.sqrt(x)
            end = time.time()
            print self.name,"  ", (end - start)

def sqr(name):
    while True:
        start = time.time()
        for i in range(1000000):
            x = 1256456
            x = np.sqrt(x)
        end = time.time()
        print name,"  ", (end - start)

    return 0


if __name__ == "__main__":
    # pool = Pool(processes = 5)
    #
    # a=['1','2','3', '4', '5']
    # result = pool.map(sqr, a)
    # pool.close()
    # pool.join()

    thread1 = myThread(1, "XX", 1)
    thread2 = myThread(2, "OO", 2)

    # Start new Threads
    thread1.start()
    thread2.start()

    thread2.join()
