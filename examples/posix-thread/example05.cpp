// =================================================================
//
// File: example05.cpp
// Author: Pedro Perez
// Description: This file contains the code that implements the
//				bubble sort algorithm. The time this implementation takes
//				will be used as the basis to calculate the improvement
//				obtained with parallel technologies.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include <pthread.h>
#include "utils.h"

const int SIZE = 10000; //1e5
const int THREADS = 4;

typedef struct {
  int start, end; // [start, end)
  int *arr, *temp;
  int step;
} Block;

using namespace std;

void swap(int *A, int i, int j) {
  int aux = A[i];
  A[i] = A[j];
  A[j] = aux;
}

void* oneStep(void *param) {
	int start;
	Block *block;

	block = (Block*) param;
	if (block->step == 0) {
		start = (block->start % 2 == 0)? block->start : (block->start + 1);
	} else {
		start = (block->start % 2 == 1)? (block->start + 1) : block->start;
	}

	for (int i = start; i < block->end; i += 2) {
		if (((i + 1) < SIZE) && block->temp[i] > block->temp[i + 1]) {
			swap(block->temp, i, i + 1);
		}
	}

	pthread_exit(0);
}

int main(int argc, char* argv[]) {
	int *a, *b, blockSize;
	double ms;
	Block blocks[THREADS];
	pthread_t tids[THREADS];

	a = new int[SIZE];
	random_array(a, SIZE);
	display_array("before", a);

	b = new int[SIZE];

	blockSize = SIZE / THREADS;
	for (int i = 0; i < THREADS; i++) {
		blocks[i].arr = a;
		blocks[i].temp = b;
		blocks[i].start = i * blockSize;
		blocks[i].end = (i != (THREADS - 1))? (i + 1) * blockSize : SIZE;
	}

	cout << "Starting..." << endl;
	ms = 0;
	for (int j = 0; j < N; j++) {
		start_timer();

		memcpy(b, a, sizeof(int) * SIZE);
		for (int step = 0; step < SIZE; step++) {
			for (int i = 0; i < THREADS; i++) {
				blocks[i].step = (step % 2 == 0);
				pthread_create(&tids[i], NULL, oneStep, (void*) &blocks[i]);
			}

			for (int i = 0; i < THREADS; i++) {
				pthread_join(tids[i], NULL);
			}
		}

		ms += stop_timer();
	}
	display_array("after", b);
	cout << "avg time = " << setprecision(5) << (ms / N) << " ms" << endl;

	delete [] a;
	delete [] b;
	return 0;
}
