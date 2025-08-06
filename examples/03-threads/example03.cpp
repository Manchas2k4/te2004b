// =================================================================
//
// File: example03.cpp
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//		by a vector using C/C++ threads. To compile:
//		g++ -o app -pthread example03.cpp
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

#define RENS 	30000
#define COLS 	30000
#define THREADS std::thread::hardware_concurrency()

// =================================================================
// Performs the multiplication of the matrix m by the vector, 
// leaving the result in the vector c.To do this, we limit the work 
// of each thread over a range of rows [start, end).
//
// @param start, the lower limit.
// @param end, the upper limit.
// @param m, a matrix of integer numbers.
// @param b, an array of integer number.
// @param c, an array of integer number.
// =================================================================
void matrix_vector(int start, int end, int *m, int *b, int *c) {
    for (int i = start; i < end; i++) {
        int acum = 0;
        for (int j = 0; j < COLS; j++) {
            acum += (m[(i * COLS) + j] * b[i]);
        }
        c[i] = acum;
    }
}

int main(int argc, char* argv[]) {
    // We will use pointers to handle large arrays.
    int *m, *b, *c;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    // We create the arrays.
    m = new int[RENS * COLS];
    b = new int [RENS];
    c = new int [RENS];

    // We fill the matrix with incremental values ​​between 1 and 
    // TOP_VALUE and the array b with one's.
    for (int i = 0; i < RENS; i++) {
        for (int j = 0; j < COLS; j++) {
            m[(i * COLS) + j] = (j + 1);
        }
        b[i] = 1;
    }

    // We calculate the block size that is assigned to each thread 
    // by dividing the task size by the number of threads.
    int blockSize = ceil((double) RENS / THREADS);
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
            int end = (i != (THREADS - 1))? ((i + 1) * blockSize) : RENS;
            threads[i] = thread(matrix_vector, start, end, m, b, c);
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
    display_array("c:", c);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    // We delete all the allocated memory.
    delete [] m;
    delete [] b;
    delete [] c;

    return 0;
}
