// =================================================================
//
// File: example03.cpp
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval
//				using pthreads.
//              To compile: g++ example3.cpp -lpthread
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <unistd.h>
#include <pthread.h>
#include "utils.h"

using namespace std;

const double PI = 3.14159265;
const int RECTS = 100000000; //1e9
const int THREADS = 4;

typedef struct {
  int start, end; // [start, end)
  double x, dx;
  double (*fn) (double);
  double result;
} Block;

void* integration(void* param) {
	int i;
	double acum;
	Block *block;

	block = (Block *) param;
	acum = 0;
	for (i = block->start; i < block->end; i++) {
		acum += block->fn(block->x + (i * block->dx));
	}
	block->result = acum * block->dx;
	pthread_exit(0);
}

int main(int argc, char* argv[]) {
	int i, j, block_size;
	double ms, result;
	Block blocks[THREADS];
	pthread_t tids[THREADS];

	block_size = RECTS / THREADS;
	for (i = 0; i < THREADS; i++) {
		blocks[i].start = i * block_size;
		blocks[i].end = (i != (THREADS - 1))? (i + 1) * block_size : RECTS;
		blocks[i].x = 0;
		blocks[i].dx = PI / RECTS;
		blocks[i].fn = sin;
		blocks[i].result = 0;
	}

	std::cout << "Starting...\n";
	ms = 0;
	for (j = 0; j < N; j++) {
		start_timer();

		for (i = 0; i < THREADS; i++) {
			pthread_create(&tids[i], NULL, integration, (void*) &blocks[i]);
		}

		result = 0;
		for (i = 0; i < THREADS; i++) {
			pthread_join(tids[i], NULL);
			result += blocks[i].result;
		}

		ms += stop_timer();
	}
	std::cout << "area = " << setprecision(5) << result << "\n";
	std::cout << "avg time =  " << setprecision(5) << (ms / N) << "\n";

	return 0;
}
