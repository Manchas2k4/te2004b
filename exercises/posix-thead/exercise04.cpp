// =================================================================
//
// File: exercise04.cpp
// Author(s):
// Description: This file implements the PI approximation using the
//				series proposed by Euler using using pthreads.
//              To compile: g++ exercise04.cpp -lpthread
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include "utils.h"
#include <omp.h>

const int LIMIT = 100000000; //1e8

using namespace std;

// implement your code here

int main(int argc, char* argv[]) {
	double result;
	double ms;

	result = 0;

	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		start_timer();

		// call your code here

		ms += stop_timer();
	}
	cout << "result = " << setprecision(40) << result << "\n";
	cout << "avg time = " << setprecision(5) << (ms / N) << " ms" << endl;

	return 0;
}
