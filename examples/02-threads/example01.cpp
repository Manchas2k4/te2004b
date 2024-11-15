// =================================================================
//
// File: example01.cpp
// Author: Pedro Perez
// Description: This file implements the addition of two vectors 
//		using C/C++ threads. To compile:
//		g++ -o app -pthread example01.cpp
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cmath>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE    1000000000 // 1e9
#define THREADS std::thread::hardware_concurrency()

// =================================================================
// Performs the sum of a segment [start, end) of two vectors, A and 
// B, leaving the result in vector C.
//
// @param start, the lower limit.
// @param end, the upper limit.
// @param a, an array of integer numbers.
// @param b, an array of integer numbers.
// @param c, an array of integer numbers.
// =================================================================
void add_vectors(int start, int end, int *a, int *b, int *c) {
    for (int i = start; i < end; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    // We will use pointers to handle large arrays.
    int *a, *b, *c;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    // We create the arrays.
    a = new int [SIZE];
    b = new int [SIZE];
    c = new int [SIZE];

    cout << "Threads = " << THREADS << "\n";

    // We fill the arrays A and B with incremental values ​​between 1 
    // and TOP_VALUE
    fill_array(a, SIZE);
    display_array("a:", a);
    fill_array(b, SIZE);
    display_array("b:", b);

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
        // We take a clock record before execution.
        start = high_resolution_clock::now();

        // We generate the threads and assign them the tasks they 
        // must perform. Each task is delimited within the range 
        // [start, end). 
        // FORK STEP
        for (int i = 0; i < THREADS; i++) {
            int start = (i * blockSize);
            int end = (i != (THREADS - 1))? ((i + 1) * blockSize) : SIZE;
            threads[i] = thread(add_vectors, start, end, a, b, c);
        }

        // We wait for all the threads to finish their assigned 
        // task. 
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
    display_array("c:", c);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    // We delete all the allocated memory.
    delete [] a;
    delete [] b;
    delete [] c;

    return 0;
}
