// =================================================================
//
// File: example5.cpp
// Author: Pedro Perez
// Description: This file contains the code that implements the
//				bubble sort algorithm using OpenMP.
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

const int SIZE = 10000; //1e4

using namespace std;

void oddEvenSort(int *arr, int size) {
    int step, i, temp;

    #pragma omp parallel shared(arr, size) private(i, temp, step)
    for (step = 0; step < size; step++) {
        if (step % 2 == 0) {
        // even index
            #pragma omp for
            for(i = 0; i <= size - 2; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                }
            }
        } else {
        // odd index
            #pragma omp for
            for(i = 1; i <= size - 2; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
	int i, *a, *aux;
	double ms;

    a = new int[SIZE];
	random_array(a, SIZE);
	display_array("before", a);

	aux = new int[SIZE];

	cout << "Starting..." << endl;
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		memcpy(aux, a, sizeof(int) * SIZE);
		oddEvenSort(aux, SIZE);

		ms += stop_timer();
	}
    display_array("after", aux);
	cout << "avg time = " << setprecision(5) << (ms / N) << " ms" << endl;

	free(a); free(aux);
	return 0;
}
