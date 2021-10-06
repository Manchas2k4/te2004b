// =================================================================
//
// File: example6.cpp
// Author: Pedro Perez
// Description: This file implements the quick sort algorithm using
//				OpenMP.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include "utils.h"

const int SIZE = 100000; //1e4

using namespace std;

void swap(int *a, int i, int j) {
	int aux = a[i];
	a[i] = a[j];
	a[j] = aux;
}

int findPivot(int *A, int low, int high) {
	int i;

	for (i = low + 1; i <= high; i++) {
		if (A[low] > A[i]) {
			return A[low];
		} else if (A[low] < A[i]){
			return A[i];
		}
	}
	return -1;
}

int makePartition(int *a, int low, int high, int pivot) {
	int i, j;

	i = low;
	j = high;
	while (i < j) {
		swap(a, i , j);
		while (a[i] < pivot) {
			i++;
		}
		while (a[j] >= pivot) {
			j--;
		}
	}
	return i;
}

void quick(int *A, int low, int high) {
	int pivot, pos;

	pivot = findPivot(A, low, high);
	if (pivot != -1) {
		pos = makePartition(A, low, high, pivot);
		#pragma omp parallel
		{
			#pragma omp task shared(A) firstprivate(low, pos)
			{
				quick(A, low, pos - 1);
			}
			#pragma omp task shared(A) firstprivate(high, pos)
			{
				quick(A, pos, high);
			}

			#pragma omp taskwait
		}
	}
}

int* quickSort(int *A, int size) {
	int *B = new int[size];

	memcpy(B, A, size * sizeof(int));
	quick(B, 0, size - 1);
	return B;
}

int main(int argc, char* argv[]) {
	int i, j, *a, *b;
	double ms;

	a = new int[SIZE];
	random_array(a, SIZE);
	display_array("before", a);

	cout << "Starting..." << endl;
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		b = quickSort(a, SIZE);

		ms += stop_timer();

		if (i != N - 1) {
			delete [] b;
		}
	}
	display_array("after", b);
	printf("avg time = %.5lf ms\n", (ms / N));

	delete [] a;
	delete [] b;
	return 0;
}
