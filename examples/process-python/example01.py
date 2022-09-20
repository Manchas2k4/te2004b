## =================================================================
##
## File: example01.py
## Author: Pedro Perez
## Description: This file contains the code that adds all the
##				elements of an integer array. The time this
##				implementation takes will be used as the basis to
##				calculate the improvement obtained with parallel
##				technologies.
##
## Copyright (c) 2022 by Tecnologico de Monterrey.
## All Rights Reserved. May be reproduced for any non-commercial
## purpose.
##
## =================================================================
import utils
import time
import multiprocessing as mp
import threading

# CORES = mp.cpu_count() 
CORES = 4
SIZE = 100000000 ##1e8

def partialSumArray(start, end, array, queue):
    acum = 0
    for i in range(start, end):
        acum += array[i]

    queue.put(acum)

if __name__ == "__main__":
    array = [0] * SIZE
    
    utils.fillArray(array)
    utils.displayArray("array", array)

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
            process = mp.Process(target=partialSumArray, args=(start, end, array, queue,))            
            processes.append(process)
            process.start()
                
        result = 0
        for i in range(CORES):
            result += queue.get()

        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)
	
    print("sum = ", result)
    print("avg time = ", (ms / utils.N), " ms")
