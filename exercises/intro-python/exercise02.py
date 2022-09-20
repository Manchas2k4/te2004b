## =================================================================
##
## File: exercise02.py
## Author(s):
## Description: This file contains the code that performs the sum of 
##				all prime numbers less than or equal to MAXIMUM. The 
##				time this implementation takes will be used as the 
##				basis to calculate the improvement obtained with 
##				parallel technologies.
##
## Copyright (c) 2022 by Tecnologico de Monterrey.
## All Rights Reserved. May be reproduced for any non-commercial
## purpose.
##
## =================================================================

import utils
import time

MAXIMUM = 1000000 ##1e6

## Place your code here

if __name__ == "__main__":
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