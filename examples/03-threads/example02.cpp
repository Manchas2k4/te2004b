// =================================================================
//
// File: example02.cpp
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//		X within the array and replaces it with Y using C/C++ threads. 
//      To compile:
//		g++ -o app -pthread example02.cpp
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <thread>
#include <cmath>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE    1000000000 //1e9
#define THREADS std::thread::hardware_concurrency()

// =================================================================
// Replaces all occurrences of the value 'x' with the value 'y'
// between a segment [start, end) of one array.
//
// @param start, the lower limit.
// @param end, the upper limit.
// @param array, an array of integer numbers.
// @param x, the value to be searched.
// @param y, the replacement value.
// =================================================================
void replace(int start, int end, int* array, int x, int y) {
    for (int i = start; i < end; i++) {
        if (array[i] == x) {
            array[i] = y;
        }
    }
}

int main(int argc, char* argv[]) {
    // We will use pointers to handle large arrays.
    int *array, *aux, x, y;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

     // We create the array and fill it withe one's.
    array = new int[SIZE];
    for (int i = 0; i < SIZE; i++) {
        array[i] = 1;
    }
    x = 1; y = -1;
    display_array("before", array);
    
    aux = new int[SIZE];

    // We calculate the block size that is assigned to each thread 
    // by dividing the task size by the number of threads.
    int blockSize = ceil((double) SIZE / THREADS);
    thread threads[THREADS];

    // We execute the task at least 10 times (N). It is necessary 
    // to do so, since it allows us to reduce the impact of the 
    // load on the operating system at the time of execution.
    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        memcpy(aux, array, sizeof(int) * SIZE);
        
        // We take a clock record before execution.
        start = high_resolution_clock::now();

        // We generate the threads and assign them the tasks they 
        // must perform. Each task is delimited within the range 
        // [start, end). 
        // FORK STEP
        for (int i = 0; i < THREADS; i++) {
            int start = i * blockSize;
            int end = (i != (THREADS - 1))? ((i + 1) * blockSize) : SIZE;
            threads[i] = thread(replace, start, end, array, x, y);
        }

        // We wait for all threads to finish their assigned task. 
        // JOIN STEP
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }

        // We take a clock record after execution. We calculate the 
        // difference between the two records. This difference is 
        // the time it took to execute the task.
        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    // We display the result and the average execution time.
    display_array("after", aux);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    // We delete all the allocated memory.
    delete [] array;
    
    return 0;
}
