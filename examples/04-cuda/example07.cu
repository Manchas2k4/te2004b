// =================================================================
//
// File: example04.cu
// Author: Pedro Perez
// Description: This file implements the merge sort algorithm using 
//				CUDA technology. To compile:
//		        !nvcc -arch=sm_75 -o app example07.cu
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

typedef unsigned int uint;

#define SIZE 	10000000 //1e7
#define THREADS	512

__global__ void gpu_merge(int *array, int *aux, uint blockSize, const uint last) {
    int id, start, mid, end, left, right, i;
    
    id = threadIdx.x;
    while (id < last) {
        start = blockSize * id;
        mid = start + (blockSize / 2) - 1;
        end = start + blockSize - 1;
        
        left = start;
        right = mid + 1;
        i = start;
        
        if (end > last) {
            end = last;
        }
        
        if (start == end || end <= mid) {
            return;
        }
        
        while (left <= mid && right <= end) {
            if (array[left] <= array[right]) {
                aux[i++] = array[left++];
            } else {
                aux[i++] = array[right++];
            }
        }
        
        while (left <= mid) {
            aux[i++] = array[left++];
        }
        
        while (right <= end) {
            aux[i++] = array[right++];
        }

        id += (blockDim.x * gridDim.x);
    }
}

void merge_sort(int *array, int *d_array, int *d_temp, uint size) {
    int *A, *B;
    uint threadCount, last;
    
    A = d_array;
    B = d_temp;
    last = size - 1;
    for (uint blockSize = 2; blockSize < (2 * size); blockSize *= 2) {
        threadCount = size / blockSize;
        if (size % blockSize > 0) {
            threadCount++;
        }
        threadCount = min(THREADS, threadCount);
        
        gpu_merge<<<1, threadCount>>>(A, B, blockSize, last);
        
        //cudaDeviceSynchronize();
        
        A = (A == d_array)? d_temp : d_array;
        B = (B == d_array)? d_temp : d_array;
    }
}

int main(int argc, char* argv[]) {
    int *array;
    int *d_array, *d_temp;
    
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    array =  new int[SIZE];
    random_array(array, SIZE);
    display_array("before", array);
    
    cudaMalloc((void**) &d_array, SIZE * sizeof(int));
    cudaMalloc((void**) &d_temp, SIZE * sizeof(int));

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        cudaMemcpy(d_array, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_temp, array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

        start = high_resolution_clock::now();

        merge_sort(array, d_array, d_temp, SIZE);
        
        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    cudaMemcpy(array, d_array, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    display_array("after", array);
    cout << "avg time = " << fixed << setprecision(3) 
        << (timeElapsed / N) <<  " ms\n";

    delete [] array;
    
    cudaFree(d_array);
    cudaFree(d_temp);

    return 0;
}