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
import multiprocessing as mp
import threading as th

# CORES = mp.cpu_count() 
CORES = 4
RECTS = 100000000 ##1e8

def partialIntegration(id, start, end, x, dx, func, results):
	acum = 0
	for i in range(start, end):
		acum += func(x + (i * dx))
	results[id] = (acum * dx)

if __name__ == "__main__":
	results = [0] * CORES
	
	x = 0
	dx = (math.pi - 0) / RECTS
	blockSize = RECTS // CORES
	
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
				end = RECTS
			t = th.Thread(target=partialIntegration, args=(i, start, end, x, dx, math.sin, results,))            
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