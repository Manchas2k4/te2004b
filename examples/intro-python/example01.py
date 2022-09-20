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

SIZE = 100000000 ##1e8

def sumArray(array):
    acum = 0
    for i in range(len(array)):
        acum += array[i]
    return acum

if __name__ == "__main__":
    array = [0] * SIZE

    utils.fillArray(array)
    utils.displayArray("array", array)

    print("Starting...")
    startTime = endTime = ms = 0
    for i in range(utils.N):
        startTime = time.time() * 1000

        result = sumArray(array)

        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)

    print("sum = ", result)
    print("avg time = ", (ms / utils.N), " ms")