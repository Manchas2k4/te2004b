## =================================================================
##
## File: example05.py
## Author: Pedro Perez
## Description: This file contains the code that implements the
##				bubble sort algorithm. The time this implementation takes
##				will be used as the basis to calculate the improvement
##				obtained with parallel technologies.
##
## Copyright (c) 2022 by Tecnologico de Monterrey.
## All Rights Reserved. May be reproduced for any non-commercial
## purpose.
##
## =================================================================
import utils
import time

SIZE = 10000 ##1e4

def swap(array, i, j):
    aux = array[i]
    array[i] = array[j]
    array[j] = aux

def bubbleSort(array):
    aux = array.copy()
    for i in range(len(aux) - 1, -1, -1):
        for j in range(0, i):
            if aux[j] > aux[j + 1]:
                swap(aux, j, j + 1)
    return aux


if __name__ == "__main__":
    array = [0] * SIZE

    utils.randomArray(array)
    utils.displayArray("array", array)

    print("Starting...")
    startTime = endTime = ms = 0
    for i in range(utils.N):
        startTime = time.time() * 1000

        result = bubbleSort(array)

        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)

    utils.displayArray("array", result)
    print("avg time = ", (ms / utils.N), " ms")