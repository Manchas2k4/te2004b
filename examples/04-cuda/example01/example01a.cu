// =================================================================
//
// File: example01a.cu
// Author: Pedro Perez
// Description: This file implements the addition of two vectors 
//				using CUDA.
//              To compile:
//		        !nvcc -arch=sm_75 -o app example1a.cu
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 1000000000 // 1e9

__global__ void add_vector(int *result, int *a, int *b) {
    result[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main(int argc, char* argv[]) {
    int *a, *b, *c;
    int *deviceA, *deviceB, *deviceC;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    a = new int [SIZE];
    b = new int [SIZE];
    c = new int [SIZE];

    cudaMalloc((void**) &deviceA, SIZE * sizeof(int));
    cudaMalloc((void**) &deviceB, SIZE * sizeof(int));
    cudaMalloc((void**) &deviceC, SIZE * sizeof(int));

    fill_array(a, SIZE);
    display_array("a:", a);
    fill_array(b, SIZE);
    display_array("b:", b);

    cudaMemcpy(deviceA, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        add_vector<<<SIZE, 1>>>(deviceC, deviceA, deviceB);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    cudaMemcpy(c, deviceC, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    display_array("c:", c);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] a;
    delete [] b;
    delete [] c;

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}
