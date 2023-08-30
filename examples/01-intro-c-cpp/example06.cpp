// =================================================================
//
// File: example06.cpp
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval.
//				The time this implementation takes will be used as
//				the basis to calculate the improvement obtained with
//				parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cmath>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define PI 3.14159265
#define RECTS 100000000 //1e8

double square(double x) {
	return x * x;
}

double integration(int rects, double x, double dx, double (*fn) (double)) {
	double acum;

	acum = 0;
	for (int i = 0; i < rects; i++) {
		acum += fn(x + (i * dx));
	}
	acum = acum * dx;
	return acum;
}

int main(int argc, char* argv[]) {
	double result, x, dx;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	x = 0;
	dx = PI / RECTS;

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		start = high_resolution_clock::now();

		result = integration(RECTS, 0, dx, sin);

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}
	cout << "result = " << fixed << setprecision(20)  << result << "\n";
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	return 0;
}
