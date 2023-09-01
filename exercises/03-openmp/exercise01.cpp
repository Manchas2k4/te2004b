// =================================================================
//
// File: exercise01.cpp
// Author(s):
// Description: This file contains the code to count the number of
//				even numbers within an array using the OpenMP 
//				technology. To compile:
//				g++ -o app -fopenmp exercise01.cpp
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

// array size
#define SIZE 1000000000

// implement your code

int main(int argc, char* argv[]) {
	int *array, result;
	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	array = new int[SIZE];
	fill_array(array, SIZE);
	display_array("array", array);

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		// call the implemented function

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
