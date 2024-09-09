// =================================================================
//
// File: example04.cpp
// Author: Pedro Perez
// Description: This file implements the algorithm to find the 
//				minimum value in an array. The time this 
//				implementation takes will be used as the basis to 
//				calculate the improvement obtained with parallel 
//				technologies.
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

#define SIZE 1000000000 // 1e9

// =================================================================
// Returns the minimum value found in an array of integers.
//
// @param array, an array of integer numbers.
// @param size, the amount of numbers.
// @return the minimum value in the array.
// =================================================================
int minimum(int *array, int size) {
    int result = array[0];
    for (int i = 0; i < size; i++) {
        if (array[i] < result) {
            result = array[i];
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    // We will use pointers to handle large arrays.
    int *array, result;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    // We create the array.
    array = new int [SIZE];
    
    // We fill it with random numbers.
    random_array(array, SIZE);
    display_array("array:", array);

    // We execute the task at least 10 times (N). It is necessary 
    // to do so, since it allows us to reduce the impact of the 
    // load on the operating system at the time of execution.
    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        // We take a clock record before execution.
        start = high_resolution_clock::now();

        // We perform the task and keep the result.
        result = minimum(array, SIZE);

        // We take a clock record after execution. We calculate the 
        // difference between the two records. This difference is 
        // the time it took to execute the task.
        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    // We display the result and the average execution time.
    cout << "result = " << result << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    // We delete all the allocated memory.
    delete [] array;

    return 0;
}
