// =================================================================
//
// File: example03.cpp
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector. The time this implementation takes will
//				be used as the basis to calculate the improvement
//				obtained with parallel technologies.
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define RENS 30000
#define COLS 30000

// =================================================================
// Performs the multiplication of the matrix m by the vector, 
// leaving the result in the vector c.
//
// @param m, a matrix of integer numbers.
// @param b, an array of integer number.
// @param c, an array of integer number.
// =================================================================
void matrix_vector(int *m, int *b, int *c) {
    int acum;

    for (int i = 0; i < RENS; i++) {
        acum = 0;
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

    // We execute the task at least 10 times (N). It is necessary 
    // to do so, since it allows us to reduce the impact of the 
    // load on the operating system at the time of execution.
    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        // We take a clock record before execution.
        start = high_resolution_clock::now();

        // We perform the task.
        matrix_vector(m, b, c);

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
