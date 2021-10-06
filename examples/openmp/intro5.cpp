// =================================================================
//
// File: intro2.cpp
// Author: Pedro Perez
// Description: This file shows the use of basic OpenMP directives.
//              To compile: g++ intro5.cpp -fopenmp
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <cstdio>
#include <iomanip>
#include <omp.h>

using namespace std;
int main(int argc, char* argv[]) {
	printf("Using the share clause\n");
	int x = 1;
	#pragma omp parallel shared(x) num_threads(3)
	{
		x++;
		printf("In the parallel block, x is %i\n", x);
	}
	printf("Outside the parallel block, x is %i\n", x);

	printf("\nUsing the private clause\n");
	x = 2;
	#pragma omp parallel private(x) num_threads(3)
	{
		x++;
		printf("In the parallel block, x is %i\n", x);
	}
	printf("Outside the parallel block, x is %i\n", x);

	printf("\nUsing the firstprivate clause\n");
	x = 2;
	#pragma omp parallel firstprivate(x) num_threads(3)
	{
		x++;
		printf("In the parallel block, x is %i\n", x);
	}
	printf("Outside the parallel block, x is %i\n", x);

	return 0;
}
