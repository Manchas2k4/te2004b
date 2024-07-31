// =================================================================
//
// File: example03.cpp
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector using CUDA. To compile
//		        nvcc -o app example03.cu
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include "utils.h"
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

#define RENS 	30000
#define COLS 	30000
#define THREADS	512
#define BLOCKS	min(4, (((RENS * COLS) / THREADS) + 1))

__global__ void matrix_vector(int *m, int *b, int *c) {
    int index, acum;

    index = threadIdx.x + (blockIdx.x * blockDim.x);

    while (index < RENS) {
        acum = 0;
        for (int j = 0; j < COLS; j++) {
            acum += (m[index * COLS] + j) * b[index];
        }
        c[index] = acum;

        //index += blockDim.x;
        index += (blockDim.x * gridDim.x); 
    }
}

int main(int argc, char* argv[]) {
    int *m, *b, *c;
    int *deviceM, *deviceB, *deviceC;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    m = new int[RENS * COLS];
    b = new int [RENS];
    c = new int [RENS];

    for (int i = 0; i < RENS; i++) {
        for (int j = 0; j < COLS; j++) {
            m[(i * COLS) + j] = (j + 1);
        }
        b[i] = 1;
    }

    cudaMalloc((void**) &deviceM, 
        (RENS * COLS) * sizeof(int));
    cudaMalloc((void**) &deviceB, RENS * sizeof(int));
    cudaMalloc((void**) &deviceC, RENS * sizeof(int));

    cudaMemcpy(deviceM, m, (RENS * COLS) * sizeof(int),
        cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b, RENS * sizeof(int),
        cudaMemcpyHostToDevice);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        matrix_vector<<<BLOCKS, THREADS>>>(deviceM,
            deviceB, deviceC);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    cudaMemcpy(c, deviceC, 
        RENS * sizeof(int), cudaMemcpyDeviceToHost);

    display_array("c:", c);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] m;
    delete [] b;
    delete [] c;

    cudaFree(deviceM);
    cudaFree(deviceB);
    cudaFree(deviceC);
    
    return 0;
}
