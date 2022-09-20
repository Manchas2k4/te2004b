// =================================================================
//
// File: example06.cpp
// Author: Pedro Perez
// Description: This file implements the quick sort algorithm. The
//				time this implementation takes will be used as the
//				basis to calculate the improvement obtained with
//				parallel technologies.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include "utils.h"

const int SIZE = 100000; //1e5

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

int makePartition(int *A, int low, int high, int pivot) {
	int i, j;

	i = low;
	j = high;
	while (i < j) {
		swap(A, i , j);
		while (A[i] < pivot) {
			i++;
		}
		while (A[j] >= pivot) {
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
		quick(A, low, pos - 1);
		quick(A, pos, high);
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
