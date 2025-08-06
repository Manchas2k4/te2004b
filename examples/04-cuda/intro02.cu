// =================================================================
//
// File: intro02.cu
// Author: Pedro Perez
// Description: This file shows some of the basic CUDA directives.
//		        To compile: nvcc -o app intro02.cu
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <cstdio>
#include <cuda_runtime.h>

using namespace std;

__global__ void kernel(void) {
    printf("GPU B%i T%i: Hello world\n", blockIdx.x, threadIdx.x);
}

int main(int argc, char* argv[]) {
    kernel<<<2, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}
