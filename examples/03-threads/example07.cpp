// =================================================================
//
// File: example07.cpp
// Author: Pedro Perez
// Description: This file shows the parallel implementation using of the
//		merge sort algorithm using C/C++ threads. To compile:
//		g++ -o app --pthread example07.cpp
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
#include <thread>
#include <cstdlib>
#include <cstring>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 	10000000 //1e7
#define THREADS std::thread::hardware_concurrency()

typedef struct {
    int *A, *B, size, index, blockSize, threadsRequired;
} Block;

void merge_task(int *A, int *B, int size, 
        int index, int block_size, int threads_required) {
            
    int start, mid, end, i, j, k;

    while (index < size) {
        start = index * block_size;
        mid = start + (block_size / 2) - 1;
        end = start + block_size - 1;
        
        i = start;
        j = mid + 1;
        k = start;
        
        if (end > (size - 1)) {
            end = size - 1;
        }
        
        if (start == end || end <= mid) {
            return;
        }
        
        while (i <= mid && j <= end) {
            if (A[i] <= A[j]) {
                B[k++] = A[i++];
            } else {
                B[k++] = A[j++];
            }
        }
        
        while (i <= mid) {
            B[k++] = A[i++];
        }
        
        while (j <= end) {
            B[k++] = A[j++];
        }

        index += threads_required;
    }
}

void parallel_merge_sort(int *array, int size) {
    int *temp, *A, *B, threads_required;
    Block *blocks; 
    thread *threads; 

    temp = new int[size];
    memcpy(temp, array, sizeof(int) * size);
    
    A = array;
    B = temp;

    for (int block_size = 2; block_size < (2 * size); block_size *= 2) {
        threads_required = min((int) (THREADS - 1), size / block_size);
        if (size % block_size > 0) {
            threads_required++;
        }
        
        blocks = new Block[threads_required];
        threads = new thread[threads_required];
        for (int i = 0; i < threads_required; i++) {
            threads[i] = thread(merge_task, A, B, size, i, block_size, threads_required);
        }

        for (int i = 0; i < threads_required; i++) {
            threads[i].join();
        }

        delete [] blocks;
        delete [] threads;
        
        A = (A == array)? temp : array;
        B = (B == array)? temp : array;
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
