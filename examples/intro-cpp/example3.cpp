// =================================================================
//
// File: example3.cpp
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
#include <cmath>
#include <omp.h>
#include "utils.h"

const double PI = 3.14159265;
const int RECTS = 100000000; //1e9

using namespace std;

double integration(double a, double b, double (*fn) (double)) {
	int i;
	double high, dx, acum, x;

	x = min(a, b);
	dx = (max(a, b) - min(a, b)) / (double) RECTS;
	acum = 0;
	#pragma omp parallel for shared(x, dx) private(i) reduction(+:acum)
	for (i = 0; i < RECTS; i++) {
		acum += fn(x + (i * dx));
	}
	return (acum * dx);
}

int main(int argc, char* argv[]) {
	int i, j;
	double ms, result;

	std::cout << "Starting...\n";
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		result = integration(0, PI, sin);

		ms += stop_timer();
	}
	std::cout << "area = " << setprecision(5) << result << "\n";
	std::cout << "avg time =  " << setprecision(5) << (ms / N) << "\n";

	return 0;
}
