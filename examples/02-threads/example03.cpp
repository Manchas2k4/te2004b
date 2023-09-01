// =================================================================
//
// File: example03.cpp
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector using POSIX threads. To compile:
//				g++ -o app -pthread example01.cpp
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

#define RENS 		30000
#define COLS 		30000
#define MAXTHREADS 	8

typedef struct {
	int *m, *b, *c;
	int start, end;
} Block;

void* matrix_vector(void* param) {
	Block *block;

	block = (Block*) param;
	for (int i = block->start; i < block->end; i++) {
		int acum = 0;
		for (int j = 0; j < COLS; j++) {
			acum += (block->m[(i * COLS) + j] * block->b[i]);
		}
		block->c[i] = acum;
	}
	return 0;
}

int main(int argc, char* argv[]) {
	int *m, *b, *c;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	int blockSize;
	Block blocks[MAXTHREADS];
	pthread_t threads[MAXTHREADS];

	m = new int[RENS * COLS];
	b = new int [RENS];
	c = new int [RENS];

	for (int i = 0; i < RENS; i++) {
		for (int j = 0; j < COLS; j++) {
			m[(i * COLS) + j] = (j + 1);
		}
		b[i] = 1;
	}

	blockSize = RENS / MAXTHREADS;
	for (int i = 0; i < MAXTHREADS; i++) {
		blocks[i].m = m;
		blocks[i].b = b;
		blocks[i].c = c;
		blocks[i].start = (i * blockSize);
		blocks[i].end = (i != (MAXTHREADS - 1))? ((i + 1) * blockSize) : RENS;
	}

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		for (int i = 0; i < MAXTHREADS; i++) {
			pthread_create(&threads[i], NULL, matrix_vector, &blocks[i]);
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

	delete [] m;
	delete [] b;
	delete [] c;

	return 0;
}
