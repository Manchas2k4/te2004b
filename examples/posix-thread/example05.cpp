// =================================================================
//
// File: example05.cpp
// Author: Pedro Perez
// Description: This file contains the code that implements the
//				bubble sort algorithm using pthreads.
//              To compile: g++ example5.cpp -lpthread
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

using namespace std;

const int SIZE = 100000; //1e5
const int THREADS = 4;

typedef struct {
    int start, end;
    int* A;
} Block;


void mergeAndCopy(int *A, int *B, int low, int mid, int high) {
    int i, j, k;
    i = low;
    j = mid;
    k = low;
    while(i < mid && j < high){
        if(A[i] < A[j]){
            B[k] = A[i];
            i++;
        } else{
            B[k] = A[j];
            j++;
        }
        k++;
    }
    for(; j < high; j++){
        B[k++] = A[j];
    }

	for(; i < mid; i++){
        B[k++] = A[i];
    }
    for (i = low; i < high; i++) {
        A[i] = B[i];
    }
}

void* partialSort(void *param) {
    Block *block;
    int aux;

    block = (Block *) param;
    for(int i = block->end - 1; i > block->start; i--){
		for(int j = block->start; j < i; j++){
			if(block->A[j] > block->A[j + 1]){
                aux = block->A[i];
                block->A[i] = block->A[j];
                block->A[j] = aux;
			}
		}
	}
    pthread_exit(NULL);
}

int* bubbleSort(int *array, int size) {
    int block_size, *A, *B, i, gap, n;
    Block blocks[THREADS];
    pthread_t tids[THREADS];

    A  = new int[size];
    B  = new int[size];

    memcpy(A, array, size * sizeof(int));

    block_size = size / THREADS;
    for (i = 0; i < THREADS; i++) {
        blocks[i].start = i * block_size;
        if (i != (THREADS - 1)) {
            blocks[i].end = (i + 1) * block_size;
        } else {
            blocks[i].end = size;
        }
        blocks[i].A = A;
    }

    for (i = 0; i < THREADS; i++) {
        pthread_create(&tids[i], NULL, partialSort, (void*) &blocks[i]);
    }

    for (i = 0; i < THREADS; i++) {
        pthread_join(tids[i], NULL);
    }

    n = THREADS;
    gap = 1;
    while (n >= 1) {
        i = 0;
        while (i < THREADS) {
            if ( (i + gap) >= THREADS ) {
                break;
            }

            mergeAndCopy(A, B, blocks[i].start, blocks[i].end, blocks[i + gap].end);
            blocks[i].end = blocks[i + gap].end;

            i += (gap * 2);
        }
        n = n / 2;
        gap = gap * 2;
    }

    delete [] B;
    return A;
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

		b = bubbleSort(a, SIZE);

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
