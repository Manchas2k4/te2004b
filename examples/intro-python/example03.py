## =================================================================
##
## File: example03.py
## Author: Pedro Perez
## Description: This file contains the code to perform the numerical
##				integration of a function within a defined interval.
##				The time this implementation takes will be used as
##				the basis to calculate the improvement obtained with
##				parallel technologies.
##
## Copyright (c) 2022 by Tecnologico de Monterrey.
## All Rights Reserved. May be reproduced for any non-commercial
## purpose.
##
## =================================================================
import utils
import time
import math

RECTS = 100000000 ##1e8

def integration(start, end, func):
    x = start
    dx = (end - start) / RECTS
    acum = 0
    for i in range(RECTS):
        acum += func(x + (i * dx))
    return (acum * dx)

if __name__ == "__main__":
    print("Starting...")
    startTime = endTime = ms = 0
    for i in range(utils.N):
        startTime = time.time() * 1000

        result = integration(0, math.pi, math.sin)

        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)

    print("area = ", result)
    print("avg time = ", (ms / utils.N), " ms")