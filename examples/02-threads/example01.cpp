// =================================================================
//
// File: example01.cpp
// Author: Pedro Perez
// Description: This file implements the addition of two vectors 
//				using POSIX threads. To compile:
//				g++ -o app -fopenmp example01.cpp
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <pthread.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 		10000000 // 1e7
#define MAXTHREADS 	8

typedef struct {
	int *c, *a, *b;
	int start, end;
} Block;

void* add_vectors(void *param) {
	Block *block;

	block = (Block*) param;
	for (int i = block->start; i < block->end; i++) {
		block->c[i] = block->a[i] + block->b[i];
	}
	return 0;
}

int main(int argc, char* argv[]) {
	int *a, *b, *c;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	int blockSize;
	Block blocks[MAXTHREADS];
	pthread_t threads[MAXTHREADS];

	a = new int [SIZE];
	b = new int [SIZE];
	c = new int [SIZE];

	fill_array(a, SIZE);
	display_array("a:", a);
	fill_array(b, SIZE);
	display_array("b:", b);

	blockSize = SIZE / MAXTHREADS;
	for (int i = 0; i < MAXTHREADS; i++) {
		blocks[i].c = c;
		blocks[i].a = a;
		blocks[i].b = b;
		blocks[i].start = (i * blockSize);
		blocks[i].end = (i != (MAXTHREADS - 1))? ((i + 1) * blockSize) : SIZE;
	}

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		for (int i = 0; i < MAXTHREADS; i++) {
			pthread_create(&threads[i], NULL, add_vectors, &blocks[i]);
		}

		for (int i = 0; i < MAXTHREADS; i++) {
			pthread_join(threads[i], NULL);
		}

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	display_array("c:", c);
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	delete [] a;
	delete [] b;
	delete [] c;

	return 0;
}
