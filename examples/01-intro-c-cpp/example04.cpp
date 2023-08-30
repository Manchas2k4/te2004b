// =================================================================
//
// File: example04.cpp
// Author: Pedro Perez
// Description: This file implements the algorithm to find the 
//				minimum value in an array. The time this 
//				implementation takes will be used as the basis to 
//				calculate the improvement obtained with parallel 
//				technologies.
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
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 1000000000 // 1e9

int minimum(int *array, int size) {
	int result = array[0];
	for (int i = 0; i < size; i++) {
		if (array[i] < result) {
			result = array[i];
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
