## =================================================================
##
## File: example03.py
## Author: Pedro Perez
## Description: This file contains the code to perform the numerical
##				integration of a function within a defined interval
##				using threads in Python.
##
## Copyright (c) 2022 by Tecnologico de Monterrey.
## All Rights Reserved. May be reproduced for any non-commercial
## purpose.
##
## =================================================================
import utils
import time
import math
import multiprocessing as mp
import threading

# CORES = mp.cpu_count() 
CORES = 4
RECTS = 100000000 ##1e8

def partialIntegration(start, end, x, dx, func, queue):
    acum = 0
    for i in range(start, end):
        acum += func(x + (i * dx))
    queue.put(acum * dx)

if __name__ == "__main__":
    blockSize = RECTS // CORES
    x = 0
    dx = (math.pi - 0) / RECTS

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
                end = RECTS
            process = mp.Process(target=partialIntegration, args=(start, end, x, dx, math.sin, queue, ))
            processes.append(process)
            process.start()
                
        result = 0
        for i in range(CORES):
            result += queue.get()

        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)

    print("area = ", result)
    print("avg time = ", (ms / utils.N), " ms")