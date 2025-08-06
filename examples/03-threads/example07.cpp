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

void merge_task(int *A, int *B, int size, int index,
                int blockSize, int threadsRequired) {
    int start, mid, end, left, right, i, numberOfThreads;

    while (index < size) {
        start = blockSize * index;
        mid = start + (blockSize / 2) - 1;
        end = start + blockSize - 1;
        
        left = start;
        right = mid + 1;
        i = start;
        
        if (end > (size - 1)) {
            end = size - 1;
        }
        
        if (start == end || end <= mid) {
            return;
        }
        
        while (left <= mid && right <= end) {
            if (A[left] <= A[right]) {
                B[i++] = A[left++];
            } else {
                B[i++] = A[right++];
            }
        }
        
        while (left <= mid) {
            B[i++] = A[left++];
        }
        
        while (right <= end) {
            B[i++] = A[right++];
        }

        index += threadsRequired;
    }
}

void parallel_merge_sort(int *array, int size) {
    int *temp, *A, *B, threadsRequired;
    thread *threads;

    temp = new int[size];
    memcpy(temp, array, sizeof(int) * size);
    
    A = array;
    B = temp;

    for (int blockSize = 2; blockSize < (2 * size); blockSize *= 2) {
        threadsRequired = min((int) THREADS, size / blockSize);
        if (size % blockSize > 0) {
            threadsRequired++;
        }

        threads = new thread[threadsRequired];
        for (int i = 0; i < threadsRequired; i++) {
            threads[i] = thread(merge_task, A, B, size, i, blockSize, threadsRequired);
        }

        for (int i = 0; i < threadsRequired; i++) {
            threads[i].join();
        }

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
