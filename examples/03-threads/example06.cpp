// =================================================================
//
// File: example06.cpp
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//		integration of a function within a defined interval 
//		using C/C++ threads. To compile:
//		g++ -o app example06.cpp
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define PI	 	3.14159265
#define RECTS 	1000000000 //1e9
#define THREADS std::thread::hardware_concurrency()

double func(double x) {
    return sin(x);
}

void integration(int start, int end, double x, double dx, 
        double (*fn) (double), double &result) {

    double acum = 0;
    for (int i = start; i < end; i++) {
        acum += fn(x + (i * dx));
    }
    result = acum;
}

int main(int argc, char* argv[]) {
    double result, x, dx, acum;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point startTime, endTime;
    double timeElapsed;

    int start, end, remainder, block_size;
    thread threads[THREADS];
    double results[THREADS];
    
    block_size = RECTS / THREADS;
    remainder = RECTS % THREADS;

    x = 0;
    dx = PI / RECTS;
    
    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        startTime = high_resolution_clock::now();

        start = 0;
        for (int i = 0; i < THREADS; i++) {
            end = start + block_size + ((i < remainder)? 1 : 0);
            threads[i] = thread(integration, start, end, x, dx,
                    func, std::ref(results[i]));
            start = end;
        }

        acum = 0;
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
            acum += results[i];
        }

        endTime = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(endTime - startTime).count();
    }
    result = acum * dx;
    cout << "result = " << fixed << setprecision(20)  << result << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    return 0;
}
