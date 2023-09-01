// =================================================================
//
// File: example03.cpp
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector using the OpenMP technology. To compile:
//				g++ -o app -fopenmp example03.cpp
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define RENS 10000
#define COLS 10000

void matrix_vector(int *m, int *b, int *c) {
	#pragma omp parallel for shared(m, b, c)
	for (int i = 0; i < RENS; i++) {
		int acum = 0;
		for (int j = 0; j < COLS; j++) {
			acum += (m[(i * COLS) + j] * b[i]);
		}
		c[i] = acum;
	}
}

int main(int argc, char* argv[]) {
	int *m, *b, *c;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	m = new int[RENS * COLS];
	b = new int [RENS];
	c = new int [RENS];

	for (int i = 0; i < RENS; i++) {
		for (int j = 0; j < COLS; j++) {
			m[(i * COLS) + j] = (j + 1);
		}
		b[i] = 1;
	}

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		matrix_vector(m, b, c);

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
