// =================================================================
//
// File: example04.cu
// Author: Pedro Perez
// Description: This file contains the code that searches for the
// 				smallest value stored in an array using CUDA. To 
//              compile:
//		        nvcc -o app example04.cu
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
#include <climits>
#include <cuda_runtime.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 1000000000 //1e9
#define THREADS 512
#define BLOCKS	min(4, ((SIZE / THREADS) + 1))

__device__ int minimum(int a, int b) {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

__global__ void minimum(int *array, int *results) {
    __shared__ int cache[THREADS];

    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int cacheIndex = threadIdx.x;

    int aux = INT_MAX;
    while (tid < SIZE) {
        aux = minimum(aux, array[tid]);
        
        tid += (blockDim.x * gridDim.x);
    }

    cache[cacheIndex] = aux;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i > 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] = minimum(cache[cacheIndex], cache[cacheIndex + i]);
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        results[blockIdx.x] = cache[cacheIndex];
    }
}

int main(int argc, char* argv[]) {
    int i, *a, *results;
    int *d_a, *d_r;
    
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    a =  new int[SIZE];
    random_array(a, SIZE);
    display_array("a", a);

    results = new int[BLOCKS];

    cudaMalloc( (void**) &d_a, SIZE * sizeof(int) );
    cudaMalloc( (void**) &d_r, BLOCKS * sizeof(int) );

    cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        minimum<<<BLOCKS, THREADS>>> (d_a, d_r);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    cudaMemcpy(results, d_r, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    int aux = INT_MAX;
    for (i = 0; i < BLOCKS; i++) {
        aux = min(aux, results[i]);
    }

    cout << "result = " << aux << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
        << (timeElapsed / N) <<  " ms\n";

    cudaFree(d_r);
    cudaFree(d_a);

    delete [] a;
    delete [] results;

    return 0;
}