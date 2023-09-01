// =================================================================
//
// File: example05.c
// Author: Pedro Perez
// Description: This file contains the approximation of Pi using the 
//				Monte-Carlo method using POSIX threads. To compile:
//				g++ -o app -pthread example05.cpp
//
// Reference:
//	https://www.geogebra.org/m/cF7RwK3H
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <random>
#include <ctime>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define INTERVAL 		 	10000 //1e4
#define NUMBER_OF_POINTS 	(INTERVAL * INTERVAL) // 1e8
#define MAXTHREADS 			8

typedef struct {
	int start, end, count;
} Block;

void* aprox_pi(void *param) {
	int local;
	Block* block;
	default_random_engine generator;
  	uniform_real_distribution<double> distribution(0.0, 1.0);

	local = 0;
	block = (Block*) param;
	for (int i = block->start; i < block->end; i++) {
		double x = (distribution(generator) * 2) - 1;
		double y = (distribution(generator) * 2) - 1;
		double dist = (x * x) + (y * y);
		if (dist <= 1) {
			local++;
		}
	}
	block->count = local;
	return 0;
}

int main(int argc, char* argv[]) {
	double result;
	int count;
	
	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	int blockSize;
	Block blocks[MAXTHREADS];
	pthread_t threads[MAXTHREADS];

	blockSize = NUMBER_OF_POINTS / MAXTHREADS;
	for (int i = 0; i < MAXTHREADS; i++) {
		blocks[i].count = 0;
		blocks[i].start = (i * blockSize);
		blocks[i].end = (i != (MAXTHREADS - 1))? ((i + 1) * blockSize) : NUMBER_OF_POINTS;
	}

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		for (int i = 0; i < MAXTHREADS; i++) {
			pthread_create(&threads[i], NULL, aprox_pi, &blocks[i]);
		}

		count = 0;
		for (int i = 0; i < MAXTHREADS; i++) {
			pthread_join(threads[i], NULL);
			count += blocks[i].count;
		}

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	cout << "count = " << count << "\n";
	result = ((double) (4.0 * count)) / ((double) NUMBER_OF_POINTS);
	cout << "result = " << fixed << setprecision(20)  << result << "\n";
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	return 0;
}
