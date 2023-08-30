// =================================================================
//
// File: example02.cpp
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//				X within the array and replaces it with Y using the 
//				OpenMP technology. To compile:
//				g++ -o app -fopenmp example02.cpp
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <omp.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 100000000 //1e8

void replace(int *array, int size, int x, int y) {
	#pragma omp parallel for shared(array, x, y, size)
	for (int i = 0; i < size; i++) {
		if (array[i] == x) {
			array[i] = y;
		}
	}
}

int main(int argc, char* argv[]) {
	int *array, *aux;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	array = new int[SIZE];
	for (int i = 0; i < SIZE; i++) {
		array[i] = 1;
	}
	display_array("before", array);
	
	aux = new int[SIZE];

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		memcpy(aux, array, sizeof(int) * SIZE);
		
		start = high_resolution_clock::now();

		replace(aux, SIZE, 1, -1);

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	
	display_array("after", aux);
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	delete [] array;
	
	return 0;
}