// =================================================================
//
// File: example01.cpp
// Author: Pedro Perez
// Description: This file implements the addition of two vectors. 
//				The time this implementation takes will be used as 
//				the basis to calculate the improvement obtained with 
//				parallel technologies.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 10000000 // 1e7

void add_vector(int *result, int *a, int *b, int size) {
	for (int i = 0; i < size; i++) {
		result[i] = a[i] + b[i];
	}
}

int main(int argc, char* argv[]) {
	int *a, *b, *c;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	a = new int [SIZE];
	b = new int [SIZE];
	c = new int [SIZE];

	fill_array(a, SIZE);
	display_array("a:", a);
	fill_array(b, SIZE);
	display_array("b:", b);

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		add_vector(c, a, b, SIZE);

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
