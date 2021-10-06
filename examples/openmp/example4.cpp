// =================================================================
//
// File: example4.cpp
// Author: Pedro Perez
// Description: This file contains the code that searches for the
// 				smallest value stored in an array using OpenMP.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <climits>
#include <algorithm>
#include <omp.h>
#include "utils.h"

const int SIZE = 1000000000; //1e9

using namespace std;

int minValue(int *array, int size) {
	int i, result;

	result = INT_MAX;
	#pragma omp parallel
	{
		int local_min = result;
		#pragma omp for nowait
		for (i = 0; i < size; i++) {
			local_min = min(local_min, array[i]);
		}

		#pragma omp critical
		{
			result = min(result, local_min);
		}
	}
	return result;
}

int main(int argc, char* argv[]) {
	int i, j, *a, pos, result;
	double ms;

	a = (int *) malloc(sizeof(int) * SIZE);
	random_array(a, SIZE);
	display_array("a", a);

	srand(time(0));
	pos = rand() % SIZE;
	printf("Setting value 0 at %i\n", pos);
	a[pos] = 0;

	cout << "Starting..." << endl;
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		result = minValue(a, SIZE);

		ms += stop_timer();
	}
	cout << "result = " << setprecision(5) << result << endl;
	cout << "avg time = " << setprecision(5) << (ms / N) << " ms" << endl;

	free(a);
	return 0;
}
