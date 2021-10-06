// =================================================================
//
// File: intro2.cpp
// Author: Pedro Perez
// Description: This file shows the use of basic OpenMP directives.
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

const int N = 7;

int main(int argc, char* argv[]) {
	int i;

	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		for (i = 0; i < N; i++) {
			printf("Thread ID %i Iteration %i\n", id ,i);
		}
	}
	return 0;
}
