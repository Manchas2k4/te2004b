// =================================================================
//
// File: example05.c
// Author: Pedro Perez
// Description: This file contains the approximation of Pi using the 
//		Monte-Carlo method using C/C++ threads. To compile:
//		g++ -o app -pthread example05.cpp
//
// Reference:
//	https://www.geogebra.org/m/cF7RwK3H
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <thread>
#include <cmath>
#include <cstring>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define INTERVAL 		 	10000 //1e4
#define NUMBER_OF_POINTS 	(INTERVAL * INTERVAL) // 1e8
#define THREADS             std::thread::hardware_concurrency()

void aprox_pi(int start, int end, int &count) {
    default_random_engine generator;
    uniform_real_distribution<double> distribution(0.0, 1.0);

    int local = 0;
    for (int i = start; i < end; i++) {
        double x = (distribution(generator) * 2) - 1;
        double y = (distribution(generator) * 2) - 1;
        double dist = (x * x) + (y * y);
        if (dist <= 1) {
            local++;
        }
    }
    count = local;
}

int main(int argc, char* argv[]) {
    double result;
    int count;
    
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    int blockSize;
    int counts[THREADS];
    thread threads[THREADS];

    memset(counts, 0, sizeof(int) * THREADS);

    blockSize = ceil((double) NUMBER_OF_POINTS / THREADS);

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        for (int i = 0; i < THREADS; i++) {
            int start = (i * blockSize);
            int end = (i != (THREADS - 1))? ((i + 1) * blockSize) : NUMBER_OF_POINTS;
            threads[i] = thread(aprox_pi, start, end, std::ref(counts[i]));
        }

        count = 0;
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
            count += counts[i];
        }

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    result = ((double) (4.0 * count)) / ((double) NUMBER_OF_POINTS);
    cout << "result = " << fixed << setprecision(20)  << result << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    return 0;
}
