// =================================================================
//
// File: example08.cpp
// Author: Pedro Perez
// Description: This file implements the merge sort algorithm. The
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
#include <chrono>
#include <cstring>
#include <cmath>
#include <pthread.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 		10000000 //1e7
#define MAX_THREADS	8

void copy_array(int *A, int *B, int low, int high) {
	int size = high - low + 1;
	memcpy(A + low, B + low, sizeof(int) * size);
}

void merge(int *A, int *B, int low, int mid, int high) {
    int i, j, k;
    i = low;
    j = mid + 1;
    k = low;
    while(i <= mid && j <= high){
        if(A[i] < A[j]){
            B[k] = A[i];
            i++;
        }else{
            B[k] = A[j];
            j++;
        }
        k++;
    }
    for(; j <= high; j++){
        B[k++] = A[j];
    }

	for(; i <= mid; i++){
        B[k++] = A[i];
    }
}

void split(int *A, int *B, int low, int high) {
    int  mid, size, i, j;

	if ((high - low + 1) == 1) {
		return;
	}

    mid = low + ((high - low) / 2);
    split(A, B, low, mid);
    split(A, B, mid +1, high);
    merge(A, B,low, mid, high);
    copy_array(A, B, low, high);
}

void merge_sort(int *A, int *B, int low, int high) {
	split(A, B, low, high);
}

typedef struct {
	int *A, *B, high, low, depth;
} Block;

void* parallel_split(void *param) {
    int  mid;
	Block *block, left, right;
	pthread_t pLeft, pRight;

	block = (Block *) param;
	if (block->depth == 0) {
		merge_sort(block->A, block->B, block->low, block->high);
	} else {
		mid = block->low + ((block->high - block->low) / 2);

		left.A = block->A;
		left.B = block->B;
		left.low = block->low;
		left.high = mid;
		left.depth = block->depth - 1;

		right.A = block->A;
		right.B = block->B;
		right.low = mid + 1;
		right.high = block->high;
		right.depth = block->depth - 1;
		
		pthread_create(&pLeft, NULL, parallel_split, &left);
		pthread_create(&pRight, NULL, parallel_split, &right);

		pthread_join(pLeft, NULL); 
		pthread_join(pRight, NULL);

		merge(block->A, block->B, block->low, mid, block->high);
		copy_array(block->A, block->B, block->low, block->high);
	}
	return 0;
}

void parallel_merge(int *A, int size) {
	int *B = new int[size];
	int depth = (int) (log(MAX_THREADS * 2) / log(2));
	Block block;
	pthread_t thread;

	block.A = A;
	block.B = B;
	block.low = 0;
	block.high = size - 1;
	block.depth = depth;

	pthread_create(&thread, NULL, parallel_split, &block);
	pthread_join(thread, NULL); 

	delete [] B;
}

int main(int argc, char* argv[]) {
	int *array, *aux;

	// These variables are used to keep track of the execution time.
	high_resolution_clock::time_point start, end;
	double timeElapsed;

	array = new int[SIZE];
	random_array(array, SIZE);
	display_array("before", array);

	aux = new int[SIZE];

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		memcpy(aux, array, sizeof(int) * SIZE);

		start = high_resolution_clock::now();

		parallel_merge(aux, SIZE);

		end = high_resolution_clock::now();
		timeElapsed += 
			duration<double, std::milli>(end - start).count();
	}

	memcpy(array, aux, sizeof(int) * SIZE);
	display_array("after", array);
	cout << "avg time = " << fixed << setprecision(3) 
		 << (timeElapsed / N) <<  " ms\n";

	delete [] array;
	delete [] aux;
	return 0;
}
