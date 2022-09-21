// =================================================================
//
// File: exercise03.cpp
// Author(s):
// Description: This file contains the code that implements the
//				enumeration sort algorithm using pthreads.
//              To compile: g++ exercise03.cpp -lpthread
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include <omp.h>
#include "utils.h"

const int SIZE = 100000; //1e5

using namespace std;

// implement your code here

int main(int argc, char* argv[]) {
	int *a, *b;
	double ms;

	a = new int[SIZE];
	random_array(a, SIZE);
	display_array("before", a);

	cout << "Starting..." << endl;
	ms = 0;
	// create object here
	for (int i = 0; i < N; i++) {
		start_timer();

		// call your code here.

		ms += stop_timer();

		if (i != N - 1) {
			delete [] b;
		}
	}
	display_array("after", b);
	cout << "avg time = " << setprecision(5) << (ms / N) << " ms" << endl;

	delete [] a;
	delete [] b;
	return 0;
}
