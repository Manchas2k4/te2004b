// =================================================================
//
// File: example05.cpp
// Author: Pedro Perez
// Description: This file contains the approximation of Pi using the 
//				Monte-Carlo method using the OpenMP technology. 
//				To compile:
//				g++ -o app -fopenmp example05.cpp
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
#include <random>
#include <cstdlib>
#include <ctime>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define INTERVAL 		 10000//1e4
#define NUMBER_OF_POINTS (INTERVAL * INTERVAL) // 1e8

double aprox_pi(int numberOfPoints) {
	int count;

	count = 0;
	#pragma omp parallel for shared(numberOfPoints) reduction(+:count)
	for (int i = 0; i < numberOfPoints; i++) {
		default_random_engine generator;
  		uniform_real_distribution<double> distribution(0.0,1.0);
		double x = (distribution(generator) * 2) - 1;
		double y = (distribution(generator) * 2) - 1;
		double dist = (x * x) + (y * y);
		if (dist <= 1) {
			count++;
		}
	}
	return ((double) (4.0 * count)) / ((double) numberOfPoints);
}

int main(int argc, char* argv[]) {
	double result;
	
	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		result = aprox_pi(NUMBER_OF_POINTS);

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	cout << "result = " << fixed << setprecision(20)  << result << "\n";
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	return 0;
}
