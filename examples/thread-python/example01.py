## =================================================================
##
## File: example01.py
## Author: Pedro Perez
## Description: This file contains the code that adds all the
##				elements of an integer array using threads in Python.
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
SIZE = 100000000 ##1e8

def partialSumArray(id, start, end, array, results):
	acum = 0
	for i in range(start, end):
		acum += array[i]
	results[id] = acum

if __name__ == "__main__":
	array = [0] * SIZE
	results = [0] * CORES
	
	utils.fillArray(array)
	utils.displayArray("array", array)

	blockSize = SIZE // CORES
	
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
				end = SIZE
			t = th.Thread(target=partialSumArray, args=(i, start, end, array, results,))            
			threads.append(t)
			t.start()
		endTime = time.time() * 1000
		ms = ms + (endTime - startTime)

		result = 0
		for i in range(CORES):
			threads[i].join()
			result += results[i]

	print("sum = ", result)
	print("avg time = ", (ms / utils.N), " ms")