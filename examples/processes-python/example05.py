## =================================================================
##
## File: example05.py
## Author: Pedro Perez
## Description: This file contains the code that implements the
##				bubble sort algorithm using processes in Python.
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
CORES = 8
SIZE = 10000 ##1e4

def swap(array, i, j):
    aux = array[i]
    array[i] = array[j]
    array[j] = aux

def bubbleSort(start, end, array):
    for i in range(end - 1, start - 1, -1):
        for j in range(start, i):
            if array[j] > array[j + 1]:
                swap(array, j, j + 1)

def mergeAndCopy(start, mid, end, array, temp):
    i = start
    j = mid
    k = start
    while i < mid and j < end:
        if array[i] < array[j]:
            temp[k] = array[i]
            i += 1
        else:
            temp[k] = array[j]
            j += 1
        k += 1

    while i < mid:
        temp[k] = array[i]
        i += 1
        k += 1

    while j < end:
        temp[k] = array[j]
        j += 1
        k += 1

    for i in range(start, end):
        array[i] = temp[i]
            
def partialSort(depth, start, end, array, temp):
    if depth == 0:
        bubbleSort(start, end, array)
    else:
        mid = start + ((end - start) // 2)
        left = mp.Process(target=partialSort, args=(depth - 1, start, mid, array, temp, ))  
        right = mp.Process(target=partialSort, args=(depth - 1, start, mid, array, temp, ))

        left.start()
        right.start()

        left.join()
        right.join()
        
        mergeAndCopy(start, mid, end, array, temp)

if __name__ == "__main__":
    array = mp.Array('i', range(SIZE))
    temp = mp.Array('i', range(SIZE))

    utils.randomArray(array)
    utils.displayArray("array", array)

    print("Starting...")
    startTime = endTime = ms = 0
    for j in range(utils.N):
        startTime = time.time() * 1000

        process = mp.Process(target=partialSort, args=(CORES, 0, SIZE, array, temp, ))  
        process.start()
        process.join()

        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)

    utils.displayArray("array", array)
    print("avg time = ", (ms / utils.N), " ms")
