// =================================================================
//
// File: example04.cpp
// Author: Pedro Perez
// Description: This file contains the code that searches for the
// 				smallest value stored in an array. The time this
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
#include <climits>
#include <algorithm>
#include "utils.h"

const int SIZE = 100000000; //1e8

using namespace std;

int minValue(int *array, int size) {
	int result = INT_MAX;
	for (int i = 0; i < size; i++) {
		result = min(result, array[i]);
	}
	return result;
}

int main(int argc, char* argv[]) {
	int *a, pos, result;
	double ms;

	a = new int[SIZE];
	random_array(a, SIZE);
	display_array("a", a);

	srand(time(0));
	pos = rand() % SIZE;
	printf("Setting value 0 at %i\n", pos);
	a[pos] = 0;

	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		start_timer();

		result = minValue(a, SIZE);

		ms += stop_timer();
	}
	cout << "result = " << setprecision(5) << result << endl;
	cout << "avg time = " << setprecision(5) << (ms / N) << " ms" << endl;

	delete [] a;
	return 0;
}
