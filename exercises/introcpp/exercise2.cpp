// =================================================================
//
// File: exercise2.cpp
// Author: Pedro Perez
// Description: This file contains the code to brute-force all
//				prime numbers less than MAXIMUM. The time this
//				implementation takes will be used as the basis to
//				calculate the improvement obtained with parallel
//				technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "utils.h"

#define MAXIMUM 1000000 //1e6

using namespace std;

// implement your code here

int main(int argc, char* argv[]) {
	int i, *a;
	double ms;

	a = new int[MAXIMUM + 1];
	memset(a, 0, (MAXIMUM + 1) * sizeof(int));
	cout << "At first, neither is a prime. We will display to TOP_VALUE:\n";
	for (i = 2; i < TOP_VALUE; i++) {
		cout << i << " ";
	}
	cout << "\n";

	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		start_timer();

		// call your code here.

		ms += stop_timer();
	}
	cout << "Expanding the numbers that are prime to TOP_VALUE:\n";
	for (i = 2; i < TOP_VALUE; i++) {
		if (a[i] == 1) {
			cout << i << " ";
		}
	}
	cout << "avg time = " << setprecision(5) << (ms / N) << " ms" << endl;

	delete [] a;
	return 0;
}
