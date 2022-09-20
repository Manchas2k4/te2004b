## =================================================================
##
## File: example02.py
## Author: Pedro Perez
## Description: This file implements the multiplication of a matrix
##				by a vector using processes in Python.
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
RENS = 10000 #1e5
COLS = 10000

def partialmatrixXVector(start, end, m, b, c):
    for i in range(start, end):
        acum = 0
        for j in range(COLS):
            acum += (m[(i * COLS) + j] * b[i])
        c[i] = acum

if __name__ == "__main__":
    m = [0] * (RENS * COLS)
    b = [0] * (RENS)
    c = mp.Array('i', range(RENS))

    for i in range(RENS):
        for j in range(COLS):
            m[(i * COLS) + j] = (j + 1)
        b[i] = 1

    blockSize = RENS // CORES

    print("Starting...")
    startTime = endTime = ms = 0
    for j in range(utils.N):
        startTime = time.time() * 1000

        processes = list()
        for i in range(CORES):
            start = i * blockSize
            if i != (CORES - 1):
                end = (i + 1) * blockSize
            else:
                end = RENS
            process = mp.Process(target=partialmatrixXVector, args=(start, end, m, b, c, ))            
            processes.append(process)
            process.start()
                
        for i in range(CORES):
            processes[i].join() 
        
        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)

    utils.displayArray("c: ", c)
    print("avg time = ", (ms / utils.N), " ms")
