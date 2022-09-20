## =================================================================
##
## File: example06.py
## Author: Pedro Perez
## Description: This file contains the code that implements the
##				bucket sort algorithm. The time this implementation 
#               takes will be used as the basis to calculate the 
##              improvement
##				obtained with parallel technologies.
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
CORES = 8
SIZE = 100000 ##1e5
BUCKETS = 200

def swap(array, i, j):
    aux = array[i]
    array[i] = array[j]
    array[j] = aux

def insertionSort(array):
    #print("entering sort")
    for i in range(1, len(array)):
        for j in range(i, 0, -1):
            if array[j] > array[j - 1]:
                break
            swap(array, j, j - 1)
    #print("ending sort")

def bucketSort(array):
    aux = array.copy()

    buckets = []
    for i in range(BUCKETS):
        buckets.append([])

    for i in range(len(array)):
        index = array[i] % BUCKETS
        buckets[index].append(array[i])

    i = BUCKETS
    while (i >= CORES):
        processes = list()
        for j in range(CORES):
            #print("\tfork j = ", (i - j - 1))
            temp = mp.Array('i', len(buckets[(i - j - 1)]))
            temp = buckets[(i - j - 1)].copy()
            process = mp.Process(target=insertionSort, args=(temp, ))  
            processes.append(process)
            process.start()
        for j in range(CORES):
            #print("\tjoin j = ", (i - j - 1))
            processes[j].join() 
            buckets[(i - j - 1)] = temp.copy()
        i = i - CORES
    
    if i > 0:
        processes = list()
        for j in range(i):
            #print("\tj = ", (i - j - 1))
            temp = mp.Array('i', len(buckets[(i - j - 1)]))
            temp = buckets[(i - j - 1)].copy()
            process = mp.Process(target=insertionSort, args=(temp, ))  
            processes.append(process)
            process.start()
        for j in range(CORES):
            #print("\tjoin j = ", (i - j - 1))
            processes[j].join()
            buckets[(i - j - 1)] = temp.copy()

    tempList = []
    for item in buckets:
        temp = sorted(tempList + item)
    
    k = 0
    for item in tempList:
        aux[k] = item
        k += 1
    
    return aux
    
if __name__ == "__main__":
    array = [0] * SIZE

    utils.randomArray(array)
    utils.displayArray("before", array)

    print("Starting...")
    startTime = endTime = ms = 0
    for i in range(utils.N):
        startTime = time.time() * 1000

        result = bucketSort(array)

        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)

    utils.displayArray("after", result)
    print("avg time = ", (ms / utils.N), " ms")