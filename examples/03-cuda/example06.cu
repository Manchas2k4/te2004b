// =================================================================
//
// File: example06.cu
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval
//				using CUDA. To compile
//		        nvcc -o app example06.cu
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
#include <cmath>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define PI 3.14159265
#define RECTS 1000000000 //1e9
#define THREADS 512
#define BLOCKS	min(4, ((RECTS / THREADS) + 1))

__device__ double square (double val) {
    return (val * val);
}

__global__ void integration(double *x, double *dx, double *results) {
    __shared__ double cache[THREADS];

    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int cacheIndex = threadIdx.x;

    double acum = 0;
    while (tid < RECTS) {
        acum += sin( (*x) + (tid * (*dx)) );
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = acum;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i > 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        results[blockIdx.x] = cache[cacheIndex];
    }
}

int main(int argc, char* argv[]) {
    double x, dx, *results;
    double *d_x, *d_dx, *d_r;
    
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    x = 0;
    dx = (PI - 0) / RECTS;

    results = new double[BLOCKS];

    cudaMalloc( (void**) &d_x, sizeof(double));
    cudaMalloc( (void**) &d_dx, sizeof(double));
    cudaMalloc( (void**) &d_r, BLOCKS * sizeof(double));

    cudaMemcpy(d_x, &x, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dx, &dx, sizeof(double), cudaMemcpyHostToDevice);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        integration<<<BLOCKS, THREADS>>> (d_x, d_dx, d_r);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    cudaMemcpy(results, d_r, BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);

    double acum = 0;
    for (int i = 0; i < BLOCKS; i++) {
        acum += results[i];
    }

    cout << "result = " << fixed << setprecision(20)  << (acum * dx) << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    cudaFree(d_x);
    cudaFree(d_dx);
    cudaFree(d_r);

    delete [] results;

    return 0;
}
