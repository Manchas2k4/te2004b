// =================================================================
//
// File: example02.cpp
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//				X within the array and replaces it with Y using CUDA. 
//              To compile:
//		        nvcc -o app example02.cu
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
#include "utils.h"
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

#define SIZE 	1000000000 //1e9
#define THREADS 512
#define BLOCKS	min(4, ((SIZE / THREADS) + 1))

__global__ void replace(int *source, int *dest, 
            int *oldElement, int* newElement) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    while (index < SIZE) {
        dest[index] = (source[index] == *oldElement)?
            *newElement : source[index];

        index += (blockDim.x * gridDim.x);
    }
}

int main(int argc, char* argv[]) {
    int *array, *aux, oldElement, newElement;
    int *deviceArray, *deviceAux;
    int *deviceOldElement, *deviceNewElement;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    array = new int[SIZE];
    for (int i = 0; i < SIZE; i++) {
        array[i] = 1;
    }
    display_array("before", array);
    
    aux = new int[SIZE];

    oldElement = 1;
    newElement = -1;

    cudaMalloc((void**) &deviceArray, SIZE * sizeof(int));
    cudaMalloc((void**) &deviceAux, SIZE * sizeof(int));
    cudaMalloc((void**) &deviceOldElement, sizeof(int));
    cudaMalloc((void**) &deviceNewElement, sizeof(int));

    cudaMemcpy(deviceArray, array, SIZE * sizeof(int), 
            cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOldElement, &oldElement, sizeof(int), 
            cudaMemcpyHostToDevice);
    cudaMemcpy(deviceNewElement, &newElement, sizeof(int), 
            cudaMemcpyHostToDevice);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        replace<<<BLOCKS, THREADS>>>(deviceArray, deviceAux, 
            deviceOldElement, deviceNewElement);

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    
    cudaMemcpy(aux, deviceAux, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    display_array("after", aux);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] array;
    delete [] aux;

    cudaFree(deviceArray);
    cudaFree(deviceAux);
    cudaFree(deviceOldElement);
    cudaFree(deviceNewElement);

    return 0;
}