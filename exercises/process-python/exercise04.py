## =================================================================
##
## File: exercise04.py
## Author(s):
## Description: This file implements the PI approximation using the
##				series proposed by Euler.
##				pi = sqrt ( 6 * sumatoria(i = 1-N) (1 / i^2) )
##				using processes in Python.
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

SIZE = 100000000 ##1e8

## Place your code here

if __name__ == "__main__":
    array = [0] * SIZE

    utils.fillArray(array)
    utils.displayArray("array", array)

    print("Starting...")
    startTime = endTime = ms = 0
	result = 0
    for i in range(utils.N):
        startTime = time.time() * 1000

        ## Place your code here

        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)

    print("result = ", result)
    print("avg time = ", (ms / utils.N), " ms")