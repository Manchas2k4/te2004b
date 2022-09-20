// =================================================================
//
// File: example06.cpp
// Author: Pedro Perez
// Description: This file implements the quick sort algorithm using
//              pthreads.
//              To compile: g++ example6.cpp -lpthread
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cstring>
#include <unistd.h>
#include <pthread.h>
#include "utils.h"

using namespace std;

const int SIZE = 100000; //1e5

typedef struct {
    int low, high;
    int *A;
} Block;

Block* makeBlock(int s, int e, int* a) {
    Block *n = new Block;
    n->low = s;
    n->high = e;
    n->A = a;
    return n;
}

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

void* quick(void* param) {
	int pivot, pos;
    Block *block, *left, *right;
    pthread_t tLeft, tRight;

    block = (Block *) param;
	pivot = findPivot(block->A, block->low, block->high);
	if (pivot != -1) {
		pos = makePartition(block->A, block->low, block->high, pivot);
        left = makeBlock(block->low, pos - 1, block->A);
        right = makeBlock(pos, block->high, block->A);

        pthread_create(&tLeft, NULL, quick, (void*) left);
        pthread_create(&tRight, NULL, quick, (void*) right);
        pthread_join(tLeft, NULL);
        pthread_join(tRight, NULL);

        delete left;
        delete right;
	}
    pthread_exit(NULL);
}

int* quickSort(int *A, int size) {
    int *B;
    Block *block;
    pthread_t tStart;

    B = new int[size];
	memcpy(B, A, size * sizeof(int));

    block = makeBlock(0, size - 1, B);
    pthread_create(&tStart, NULL, quick, (void*) block);
    pthread_join(tStart, NULL);
    delete block;

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
