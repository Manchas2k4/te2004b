## =================================================================
##
## File: example02.py
## Author: Pedro Perez
## Description: This file implements the multiplication of a matrix
##				by a vector. The time this implementation takes will
##				be used as the basis to calculate the improvement
##				obtained with parallel technologies.
##
## Copyright (c) 2022 by Tecnologico de Monterrey.
## All Rights Reserved. May be reproduced for any non-commercial
## purpose.
##
## =================================================================
import utils
import time

RENS = 5000
COLS = 5000

def matrixXVector(matrix, b, c):
    for i in range(RENS):
        acum = 0
        for j in range(COLS):
            acum += (matrix[(i * COLS) + j] * b[i])
        c[i] = acum

if __name__ == "__main__":
    matrix = [0] * (RENS * COLS)
    b = [0] * (RENS)
    c = [0] * (RENS)

    for i in range(RENS):
        for j in range(COLS):
            matrix[(i * COLS) + j] = (j + 1)
        b[i] = 1

    print("Starting...")
    startTime = endTime = ms = 0
    for i in range(utils.N):
        startTime = time.time() * 1000

        matrixXVector(matrix, b, c)

        endTime = time.time() * 1000

        ms = ms + (endTime - startTime)

    utils.displayArray("c: ", c)
    print("avg time = ", (ms / utils.N), " ms")