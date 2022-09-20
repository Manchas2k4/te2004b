## =================================================================
##
## File: example05.py
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
from heapq import merge

SIZE = 100000 ##1e5
BUCKETS = 200

def swap(array, i, j):
    aux = array[i]
    array[i] = array[j]
    array[j] = aux

def insertionSort(array):
    for i in range(len(array)):
        for j in range(i, 0, -1):
            if array[j] > array[j - 1]:
                break
            swap(array, j, j - 1)

def mergeAndCopy(arr1, arr2):
    i = 0
    j = 0
    k = 0
    while i < len(arr1) and j < len(arr2):
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

def bucketSort(array):
    aux = array.copy()

    buckets = []
    for i in range(BUCKETS):
        buckets.append([])
    
    for i in range(len(array)):
        index = array[i] % BUCKETS
        buckets[index].append(array[i])

    for i in range(BUCKETS):
        insertionSort(buckets[i])

    temp = []
    for item in buckets:
        temp = sorted(temp + item)
    
    k = 0
    for item in temp:
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