## =================================================================
##
## Author: Pedro Perez
## Description: This file contains the code that searches for the
## 				smallest value stored in an array. The time this
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
import sys
from random import seed
from random import random
from random import randrange

SIZE = 100000000 ##1e8

def minArray(array):
    result = sys.maxsize
    for i in range(len(array)):
        result = min(result, array[i])
    return result

if __name__ == "__main__":
    array = [0] * SIZE

    utils.randomArray(array)
    utils.displayArray("array", array)

    pos = randrange(SIZE)
    array[pos] = 0

    print("Starting...")
    startTime = endTime = ms = 0
    for i in range(utils.N):
        startTime = time.time() * 1000

        result = minArray(array)

        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)

    print("minimum = ", result)
    print("avg time = ", (ms / utils.N), " ms")