// =================================================================
//
// File: example1.cpp
// Author: Pedro Perez
// Description: This file contains the code that adds all the
//				elements of an integer array. The time this
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
#include "utils.h"

const int SIZE = 1000000000; //1e9

double sumArray(int *array, int size) {
	double acum;
	int i;

	acum = 0;
	for (i = 0; i < size; i++) {
		acum += array[i];
	}
	return acum;
}

int main(int argc, char* argv[]) {
	int i, j, *a;
	double ms, result;

	a = new int[SIZE];
	fill_array(a, SIZE);
	display_array("a", a);

	std::cout << "Starting...\n";
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		result = sumArray(a, SIZE);

		ms += stop_timer();
	}
	std::cout << "sum = " << setprecision(5) << result << "\n";
	std::cout << "avg time =  " << setprecision(5) << (ms / N) << "\n";

	delete [] a;
	return 0;
}
