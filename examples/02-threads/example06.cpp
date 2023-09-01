// =================================================================
//
// File: example06.cpp
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval 
//				using POSIX threads. To compile:
//				g++ -o app -pthread example06.cpp
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <pthread.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define PI	 		3.14159265
#define RECTS 		100000000 //1e8
#define MAXTHREADS 	8

double square(double x) {
	return x * x;
}

typedef struct {
	double x, dx, result;
	double (*fn) (double);
	int start, end;
} Block;

void* integration(void *param) {
	Block *block;
	double acum;

	acum = 0;
	block = (Block*) param;
	for (int i = block->start; i < block->end; i++) {
		acum += block->fn(block->x + (i * block->dx));
	}
	block->result = acum;
	return 0;
}

int main(int argc, char* argv[]) {
	double result, x, dx, acum;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	int blockSize;
	Block blocks[MAXTHREADS];
	pthread_t threads[MAXTHREADS];

	x = 0;
	dx = PI / RECTS;

	blockSize = RECTS / MAXTHREADS;
	for (int i = 0; i < MAXTHREADS; i++) {
		blocks[i].x = x;
		blocks[i].dx = dx;
		blocks[i].fn = sin;
		blocks[i].result = 0;
		blocks[i].start = (i * blockSize);
		blocks[i].end = (i != (MAXTHREADS - 1))? ((i + 1) * blockSize) : RECTS;
	}	

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		for (int i = 0; i < MAXTHREADS; i++) {
			pthread_create(&threads[i], NULL, integration, &blocks[i]);
		}

		acum = 0;
		for (int i = 0; i < MAXTHREADS; i++) {
			pthread_join(threads[i], NULL);
			acum += blocks[i].result;
		}

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	result = acum * dx;
	cout << "result = " << fixed << setprecision(20)  << result << "\n";
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	return 0;
}
