// =================================================================
//
// File: intro2.cpp
// Author: Pedro Perez
// Description: This file shows the use of basic OpenMP directives.
//              To compile: g++ intro2.cpp -fopenmp
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[]) {
	int i = 0;
	#pragma omp parallel
	{
		i++;
		int total = omp_get_num_threads();
		int iam = omp_get_thread_num();
		printf("Hello world!! I am the thread %i from %i threads ==> i = %i\n", iam, total, i);
	}
	cout << "i = " << i << "\n";
	return 0;
}
