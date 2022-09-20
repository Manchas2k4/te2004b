## =================================================================
##
## File: example04.py
## Author: Pedro Perez
## Description: This file contains the code that searches for the
## 				smallest value stored in an array using processes
##              in Python.
##
## Copyright (c) 2022 by Tecnologico de Monterrey.
## All Rights Reserved. May be reproduced for any non-commercial
## purpose.
##
## =================================================================

import utils
import time
import sys
from random import seed
from random import random
import multiprocessing as mp
import threading
from random import randrange

# CORES = mp.cpu_count() 
CORES = 4
SIZE = 100000000 ##1e8

def partialMinArray(start, end, array, queue, ):
    result = sys.maxsize
    for i in range(start, end):
        result = min(result, array[i])
    queue.put(result)

if __name__ == "__main__":
    array = [0] * SIZE

    utils.randomArray(array)
    utils.displayArray("array", array)

    pos = randrange(SIZE)
    array[pos] = 0

    blockSize = SIZE // CORES

    print("Starting...")
    startTime = endTime = ms = 0
    for j in range(utils.N):
        startTime = time.time() * 1000

        queue = mp.SimpleQueue()
        processes = list()
        for i in range(CORES):
            start = i * blockSize
            if i != (CORES - 1):
                end = (i + 1) * blockSize
            else:
                end = SIZE
            process = mp.Process(target=partialMinArray, args=(start, end, array, queue,))            
            processes.append(process)
            process.start()
                
        result = 0
        for i in range(CORES):
            result = min(result, queue.get())

        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)

    print("minimum = ", result)
    print("avg time = ", (ms / utils.N), " ms")