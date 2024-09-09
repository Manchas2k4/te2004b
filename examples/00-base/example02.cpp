// =================================================================
//
// File: example02.cpp
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//				X within the array and replaces it with Y. The time 
//				it takes to implement this will be used as the basis 
//				for calculating the improvement obtained with parallel 
//				technologies. The time this implementation takes.
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 1000000000 //1e9

// =================================================================
// Replaces all occurrences of the value 'x' with the value 'y'.
//
// @param array, an array of integer numbers.
// @param size, the amount of numbers.
// @param x, the value to be searched.
// @param y, the replacement value.
// =================================================================
void replace(int *array, int size, int x, int y) {
    int i;

    for (i = 0; i < size; i++) {
        if (array[i] == x) {
            array[i] = y;
        }
    }
}

int main(int argc, char* argv[]) {
    // We will use pointers to handle large arrays.
    int *array, *aux;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    // We create the array and fill it withe one's.
    array = new int[SIZE];
    for (int i = 0; i < SIZE; i++) {
        array[i] = 1;
    }
    display_array("before", array);
    
    aux = new int[SIZE];

    // We execute the task at least 10 times (NO). It is necessary 
    // to do so, since it allows us to reduce the impact of the 
    // load on the operating system at the time of execution.
    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        memcpy(aux, array, sizeof(int) * SIZE);
        // We take a clock record before execution.
        start = high_resolution_clock::now();

        // We perform the task.
        replace(aux, SIZE, 1, -1);

        // We take a clock record after execution. We calculate the 
        // difference between the two records. This difference is 
        // the time it took to execute the task.
        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    // We display the result and the average execution time.
    display_array("after", aux);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    // We delete all the allocated memory.
    delete [] array;
    delete [] aux;
    
    return 0;
}