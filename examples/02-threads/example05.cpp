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

// =================================================================
// Calculate the approximation of Pi using the Monte Carlo method. 
// Each thread is assigned a number of random points to generate 
// [start, end]. The value obtained is assigned to the variable 
// 'result'.
//
// @param start, the lower limit.
// @param end, the upper limit.
// @param result, to this variable is assigned the result of the 
//                operation.
// =================================================================
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

    // We calculate the block size that is assigned to each thread 
    // by dividing the task size by the number of threads.
    int blockSize = ceil((double) SIZE / THREADS);
    thread threads[THREADS];

    // We use the 'counts' array to store the results generated by 
    // each thread.
    int counts[THREADS];

    // We initialize each location in the array to zero.
    memset(counts, 0, sizeof(int) * THREADS);

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
            int end = (i != (THREADS - 1))? ((i + 1) * blockSize) : NUMBER_OF_POINTS;
            threads[i] = thread(aprox_pi, start, end, std::ref(counts[i]));
        }

        // We wait for all the threads to finish their assigned 
        // task. After a particular thread has finished, we retrieve 
        // its partial result and use it to calculate the overall 
        // result.
        // JOIN STEP
        count = 0;
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
            count += counts[i];
        }

        // We take a clock record after execution. We calculate the 
        // difference between the two records. This difference is 
        // the time it took to execute the task.
        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    // We display the result and the average execution time.
    result = ((double) (4.0 * count)) / ((double) NUMBER_OF_POINTS);
    cout << "result = " << fixed << setprecision(20)  << result << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    return 0;
}
