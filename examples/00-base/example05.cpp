// =================================================================
//
// File: example05.cpp
// Author: Pedro Perez
// Description: This file contains the approximation of Pi using the 
//				Monte-Carlo method.The time this implementation 
//				takes will be used as the basis to calculate the 
//				improvement obtained with parallel technologies.
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
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define INTERVAL 		 10000//1e4
#define NUMBER_OF_POINTS (INTERVAL * INTERVAL) // 1e8

// =================================================================
// Returns the approximation of Pi using the Monte-Carlo method.
//
// @param numberOfPoints, the number of point to be generated.
// @return the aproximation of Pi.
// =================================================================
double aprox_pi(int numberOfPoints) {
    default_random_engine generator;
    uniform_real_distribution<double> distribution(0.0, 1.0);
    int count;

    count = 0;
    for (int i = 0; i < numberOfPoints; i++) {
        double x = (distribution(generator) * 2) - 1;
        double y = (distribution(generator) * 2) - 1;
        double dist = (x * x) + (y * y);
        if (dist <= 1) {
            count++;
        }
    }
    return ((double) (4.0 * count)) / ((double) numberOfPoints);
}

int main(int argc, char* argv[]) {
    double result;
    
    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    // We execute the task at least 10 times (N). It is necessary 
    // to do so, since it allows us to reduce the impact of the 
    // load on the operating system at the time of execution.
    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        // We take a clock record before execution.
        start = high_resolution_clock::now();

        // We perform the task.
        result = aprox_pi(NUMBER_OF_POINTS);

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
