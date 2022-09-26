## =================================================================
##
## File: exercise03.py
## Author(s):
## Description: This file contains the code that implements the
##				enumeration sort algorithm using processes in Python.
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

SIZE = 10000 ##1e5

## Place your code here

if __name__ == "__main__":
    array = [0] * SIZE

    utils.randomArray(array)
    utils.displayArray("before", array)

    print("Starting...")
    startTime = endTime = ms = 0
	result = 0
    for i in range(utils.N):
        startTime = time.time() * 1000

        ## Place your code here

        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)

    utils.displayArray("after", array)
    print("avg time = ", (ms / utils.N), " ms")