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

void add_vectors(int start, int end, int *a, int *b, int *c) {
    for (int i = start; i < end; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    int *a, *b, *c;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    int blockSize;
    thread threads[THREADS];

    a = new int [SIZE];
    b = new int [SIZE];
    c = new int [SIZE];

    fill_array(a, SIZE);
    display_array("a:", a);
    fill_array(b, SIZE);
    display_array("b:", b);

    blockSize = ceil((double) SIZE / THREADS);
    
    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        for (int i = 0; i < THREADS; i++) {
            int start = (i * blockSize);
            int end = (i != (THREADS - 1))? ((i + 1) * blockSize) : SIZE;
            threads[i] = thread(add_vectors, start, end, a, b, c);
        }

        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    display_array("c:", c);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] a;
    delete [] b;
    delete [] c;

    return 0;
}
