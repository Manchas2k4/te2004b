// =================================================================
//
// File: example04.cpp
// Author: Pedro Perez
// Description: This file implements the algorithm to find the 
//				minimum value in an array using POSIX threads. To 
//				compile:
//				g++ -o app -pthread example04.cpp
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <climits>
#include <pthread.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 		1000000000 // 1e9
#define MAXTHREADS 	8

typedef struct {
	int *array, result;
	int start, end;
} Block;

void* minimum_value(void *param) {
	Block *block;
	int local;

	local = INT_MAX;
	block = (Block*) param;
	for (int i = block->start; i < block->end; i++) {
		if (block->array[i] < local) {
			local = block->array[i];
		}
	}
	block->result = local;
	return 0;
}

int main(int argc, char* argv[]) {
	int *array, result;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	int blockSize;
	Block blocks[MAXTHREADS];
	pthread_t threads[MAXTHREADS];

	array = new int [SIZE];
	
	random_array(array, SIZE);
	display_array("array:", array);

	blockSize = SIZE / MAXTHREADS;
	for (int i = 0; i < MAXTHREADS; i++) {
		blocks[i].array = array;
		blocks[i].result = INT_MAX;
		blocks[i].start = (i * blockSize);
		blocks[i].end = (i != (MAXTHREADS - 1))? ((i + 1) * blockSize) : SIZE;
	}

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		for (int i = 0; i < MAXTHREADS; i++) {
			pthread_create(&threads[i], NULL, minimum_value, &blocks[i]);
		}

		result = INT_MAX;
		for (int i = 0; i < MAXTHREADS; i++) {
			pthread_join(threads[i], NULL);
			if (blocks[i].result < result) {
				result = blocks[i].result;
			}
		}

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	cout << "result = " << result << "\n";
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	delete [] array;

	return 0;
}
