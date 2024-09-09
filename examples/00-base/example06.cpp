// =================================================================
//
// File: example06.cpp
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval.
//				The time this implementation takes will be used as
//				the basis to calculate the improvement obtained with
//				parallel technologies.
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
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define PI 3.14159265
#define RECTS 1000000000 //1e9

double square(double x) {
    return x * x;
}

// =================================================================
// Returns the the numerical integration of a function within a 
// defined interval.
//
// @param rects, the number of rectangles that will be used to 
//               calculate the integral.
// @param x, the lower limit of the interval.
// @param dx, the increase in x.
// @param fn, the functio to be integrate.
// @return an approximation of the area under the curve.
// =================================================================
double integration(int rects, double x, double dx, double (*fn) (double)) {
    double acum;

    acum = 0;
    for (int i = 0; i < rects; i++) {
        acum += fn(x + (i * dx));
    }
    acum = acum * dx;
    return acum;
}

int main(int argc, char* argv[]) {
    double result, x, dx;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    x = 0;
    dx = PI / RECTS;

    // We execute the task at least 10 times (N). It is necessary 
    // to do so, since it allows us to reduce the impact of the 
    // load on the operating system at the time of execution.
    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        // We take a clock record before execution.
        start = high_resolution_clock::now();

        // We perform the task.
        result = integration(RECTS, 0, dx, sin);

        // We take a clock record after execution. We calculate the 
        // difference between the two records. This difference is 
        // the time it took to execute the task.
        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    // We display the result and the average execution time.
    cout << "result = " << fixed << setprecision(20)  << result << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    return 0;
}
