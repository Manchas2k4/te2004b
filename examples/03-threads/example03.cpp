// =================================================================
//
// File: example03.cpp
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//		by a vector using C/C++ threads. To compile:
//		g++ -o app example03.cpp
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
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define RENS 	30000
#define COLS 	30000
#define THREADS std::thread::hardware_concurrency()

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
    int *m, *b, *c;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point startTime, endTime;
    double timeElapsed;

    int start, end, remainder, block_size;
    thread threads[THREADS];
    
    block_size = RENS / THREADS;
    remainder = RENS % THREADS;

    m = new int[RENS * COLS];
    b = new int [RENS];
    c = new int [RENS];

    for (int i = 0; i < RENS; i++) {
        for (int j = 0; j < COLS; j++) {
            m[(i * COLS) + j] = (j + 1);
        }
        b[i] = 1;
    }
    
    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        startTime = high_resolution_clock::now();

        start = 0;
        for (int i = 0; i < THREADS; i++) {
            end = start + block_size + ((i < remainder)? 1 : 0);
            threads[i] = thread(matrix_vector, start, end, m, b, c);
            start = end;
        }

        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }

        endTime = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(endTime - startTime).count();
    }
    display_array("c:", c);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] m;
    delete [] b;
    delete [] c;

    return 0;
}
