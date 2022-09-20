## =================================================================
##
## File: exercise01.py
## Author(s):
## Description: This file contains the code to count the number of
##				even numbers within an array. The time this implementation
##				takes will be used as the basis to calculate the
##				improvement obtained with parallel technologies.
##
## Copyright (c) 2022 by Tecnologico de Monterrey.
## All Rights Reserved. May be reproduced for any non-commercial
## purpose.
##
## =================================================================

import utils
import time

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

    print("sum = ", result)
    print("avg time = ", (ms / utils.N), " ms")