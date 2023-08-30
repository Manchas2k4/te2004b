// =================================================================
//
// File: example04.cpp
// Author: Pedro Perez
// Description: This file implements the algorithm to find the 
//				minimum value in an arrayusing the OpenMP 
//				technology. To compile:
//				g++ -o app -fopenmp example04.cpp
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <climits>
#include <omp.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 1000000000 // 1e9

int minimum(int *array, int size) {
	int i, result;

	result = INT_MAX;
	#pragma omp parallel
	{
		int local_min = result;
		#pragma omp for nowait
		for (i = 0; i < size; i++) {
			if (array[i] < local_min) {
				local_min = array[i];
			}
		}

		#pragma omp critical
		{
			if (local_min < result) {
				result = local_min;
			}
		}
	}
	return result;
}

int main(int argc, char* argv[]) {
	int *array, result;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	array = new int [SIZE];
	
	random_array(array, SIZE);
	display_array("array:", array);

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		result = minimum(array, SIZE);

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
