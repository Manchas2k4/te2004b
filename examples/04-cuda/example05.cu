// =================================================================
//
// File: example05.cu
// Author: Pedro Perez
// Description: This file contains the approximation of Pi using the 
//				Monte-Carlo method using CUDA. To compile
//		        !nvcc -arch=sm_75 -o app example05.cu
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
#include <curand_kernel.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define INTERVAL 		 	10000 //1e4
#define NUMBER_OF_POINTS 	(INTERVAL * INTERVAL) 
#define THREADS             512
#define BLOCKS	            min(4, ((NUMBER_OF_POINTS / THREADS) + 1))

__global__ void aprox_pi(curandState *states, int *results) {
    __shared__ int cache[THREADS];

    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int cacheIndex = threadIdx.x;
    double x, y, dist;
    int local;

    local = 0;
    while (tid < NUMBER_OF_POINTS) {
        x = (curand_uniform(&(states[cacheIndex])) * 2) - 1;
        y = (curand_uniform(&(states[cacheIndex])) * 2) - 1;
        dist = (x * x) + (y * y);
        if (dist <= 1) {
            local++;
        }
        
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = local;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i > 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        results[blockIdx.x] = cache[cacheIndex];
    }
}

__global__ void setup_random_generator(curandState *state)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(tid, tid, 0, &state[tid]);
}

int main(int argc, char* argv[]) {
    int *results, *d_results, totalThreads;
    double count, result;
    curandState *d_states;
    
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    results = new int[BLOCKS];

    totalThreads = THREADS * BLOCKS;
    cudaMalloc( (void**) &d_results, BLOCKS * sizeof(int) );
    cudaMalloc( (void**) &d_states, totalThreads * sizeof(curandState) );

    setup_random_generator<<<BLOCKS, THREADS>>> (d_states);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        aprox_pi<<<BLOCKS, THREADS>>> (d_states, d_results);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }

    cudaMemcpy(results, d_results, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

    count = 0;
    for (int i = 0; i < BLOCKS; i++) {
        count += results[i];
    }

    result = ((double) (4.0 * count)) / ((double) NUMBER_OF_POINTS);
    cout << "result = " << fixed << setprecision(20) 
        << result << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
        << (timeElapsed / N) <<  " ms\n";

    cudaFree(d_results);
    cudaFree(d_states);

    delete [] results;

    return 0;
}