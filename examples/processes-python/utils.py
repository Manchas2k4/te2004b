## =================================================================
##
## File: utils.h
## Author: Pedro Perez
## Description: This file contains the implementation of the
##				functions used to take the time and perform the
##				speed up calculation; as well as functions for
##				initializing integer arrays.
##
## Copyright (c) 2022 by Tecnologico de Monterrey.
## All Rights Reserved. May be reproduced for any non-commercial
## purpose.
##
## =================================================================

from random import seed
from random import randrange

N = 10
DISPLAY = 100
TOP_VALUE = 10000

## =================================================================
## Initializes an array with random values between 1 and TOP_VALUE.
##
## @param array, an array of integer numbers.
## =================================================================
def randomArray(array):
	for i in range(len(array)):
		array[i] = randrange(1, TOP_VALUE)


## =================================================================
## Initializes an array with consecutive values of 1 and TOP_VALUE
## across all locations.
##
## @param array, an array of integer numbers.
## =================================================================
def fillArray(array):
	for i in range(len(array)):
		array[i] = (i % TOP_VALUE) + 1


## =================================================================
## Displays the first N locations in the array.
##
## @param array, an array of integer numbers.
## =================================================================
def displayArray(text, array):
    print(text, " = [", end = " ")
    for i in range(DISPLAY):
        if i < (DISPLAY - 1):
            print(array[i], end=", ")
        else:
            print(array[i], end= "")
    print("]")
