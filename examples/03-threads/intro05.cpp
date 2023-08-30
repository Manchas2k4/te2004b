// =================================================================
//
// File: intro05.cpp
// Author: Pedro Perez
// Description: This file shows some of the basic OpenMP directives.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <omp.h>

#define N 8

int main(int argc, char* argv[]) {
	int i;

	#pragma omp parallel for num_threads(3)
	for (i = 0; i < N; i++) {
		int id = omp_get_thread_num();
		printf("Thread ID %i Iteration %i\n", id ,i);
	}
	return 0;
}
