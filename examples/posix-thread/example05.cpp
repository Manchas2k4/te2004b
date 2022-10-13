// =================================================================
//
// File: example05.cpp
// Author: Pedro Perez
// Description: This file contains the code that implements the
//				bubble sort algorithm. The time this implementation takes
//				will be used as the basis to calculate the improvement
//				obtained with parallel technologies.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include <pthread.h>
#include "utils.h"

const int SIZE = 10000; //1e5
const int THREADS = 4;

typedef struct {
  int start, end; // [start, end)
  int *arr;
} Block;

using namespace std;

void swap(int *A, int i, int j) {
  int aux = A[i];
  A[i] = A[j];
  A[j] = aux;
}

int* oddEvenSort(int *A, int size) {
    int *B = new int[size];

    memcpy(B, A, sizeof(int) * size);
	for (int step = 0; step < size; step++) {
		if (step % 2 == 0) {
			for (int i = 0; i <= size - 2; i += 2) {
				if (B[i] > B[i + 1]) {
					swap(B, i, i + 1);
				}
			}
		} else {
			for (int i = 1; i <= size - 2; i += 2) {
				if (B[i] > B[i + 1]) {
					swap(B, i, i + 1);
				}
			}
		}
	}
    return B;
}

int main(int argc, char* argv[]) {
	int *a, *b;
	double ms;

	a = new int[SIZE];
	random_array(a, SIZE);
	display_array("before", a);

	cout << "Starting..." << endl;
	ms = 0;
	for (int i = 0; i < N; i++) {
		start_timer();

		b = oddEvenSort(a, SIZE);

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
