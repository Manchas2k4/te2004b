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
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 	10000000 //1e7
#define THREADS	8

void merge(int *A, int *B, int size, int index, int blockSize, int threadsRequired) {
	int start, mid, end, left, right, i, numberOfThreads;

	while (index < size) {
		start = blockSize * index;
		mid = start + (blockSize / 2) - 1;
		end = start + blockSize - 1;
		
		left = start;
		right = mid + 1;
		i = start;
		
		if (end > (size - 1)) {
			end = size - 1;
		}
		
		if (start == end || end <= mid) {
			return;
		}
		
		while (left <= mid && right <= end) {
			if (A[left] <= A[right]) {
				B[i++] = A[left++];
			} else {
				B[i++] = A[right++];
			}
		}
		
		while (left <= mid) {
			B[i++] = A[left++];
		}
		
		while (right <= end) {
			B[i++] = A[right++];
		}

		index += threadsRequired;
	}
}

void parallel_merge_sort(int *array, int size) {
	int *temp, *A, *B, threadsRequired;

	temp = new int[size];
	memcpy(temp, array, sizeof(int) * size);
	
	A = array;
	B = temp;

	for (int blockSize = 2; blockSize < (2 * size); blockSize *= 2) {
		threadsRequired = min(THREADS, size / blockSize);
		if (size % blockSize > 0) {
			threadsRequired++;
		}
		
		#pragma omp parallel for num_threads(threadsRequired)
		for (int i = 0; i < threadsRequired; i++) {
			merge(A, B, SIZE, i, blockSize, threadsRequired);
		}
		
		#pragma omp taskwait
		{
			A = (A == array)? temp : array;
			B = (B == array)? temp : array;
		}
	}
	
	delete [] temp;
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
	
	/*
	parallel_merge_sort(array, SIZE);
	display_array("after", array);
	
	
	cout << "THREADS = " << THREADS << "\n";
	for (int i = 0; i < 4; i++) {
		merge(array, aux, SIZE, i, 2);
	}

	display_array("after", array);
	display_array("after", aux);
	*/

	cout << "Starting...\n";
	timeElapsed = 0;
	for (int j = 0; j < N; j++) {
		memcpy(aux, array, sizeof(int) * SIZE);

		start = high_resolution_clock::now();

		parallel_merge_sort(aux, SIZE);

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
