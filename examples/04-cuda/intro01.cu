// =================================================================
//
// File: intro01.cu
// Author: Pedro Perez
// Description: This file shows some of the basic CUDA directives.
//		        To compile: !nvcc -arch=sm_75 -o app intro01.cu
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main(int argc, char* argv[]) {
    int a, b, c;
    int *d_a, *d_b, *d_c;
    
    a = 12;
    b = 13;

    cudaMalloc((void**) &d_a, sizeof(int));
    cudaMalloc((void**) &d_b, sizeof(int));
    cudaMalloc((void**) &d_c, sizeof(int));

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    add<<<1, 1>>>(d_a, d_b, d_c);

    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "c = " << c << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
