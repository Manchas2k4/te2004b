## =================================================================
##
## File: example02.py
## Author: Pedro Perez
## Description: This file implements the multiplication of a matrix
##				by a vector using threads in Python.
##
## Copyright (c) 2022 by Tecnologico de Monterrey.
## All Rights Reserved. May be reproduced for any non-commercial
## purpose.
##
## =================================================================
import utils
import time
import multiprocessing as mp
import threading as th

# CORES = mp.cpu_count() 
CORES = 4
RENS = 5000
COLS = 5000

def partialMatrixXVector(start, end, matrix, b, c):
	for i in range(start, end):
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

	blockSize = RENS // CORES

	print("Starting...")
	startTime = endTime = ms = 0
	for i in range(utils.N):
		startTime = time.time() * 1000
		
		threads = list()
		for i in range(CORES):
			start = i * blockSize
			if i != (CORES - 1):
				end = (i + 1) * blockSize
			else:
				end = RENS
			t = th.Thread(target=partialMatrixXVector, args=(start, end, matrix, b, c,))            
			threads.append(t)
			t.start()
		endTime = time.time() * 1000
		ms = ms + (endTime - startTime)

		for i in range(CORES):
			threads[i].join()

	utils.displayArray("c: ", c)
	print("avg time = ", (ms / utils.N), " ms")