// =================================================================
//
// File: example02.cpp
// Author: Pedro Perez
// Description: This file contains the code that looks for an element 
//		X within the array and replaces it with Y using C/C++ threads. 
//      To compile:
//		g++ -o app example02.cpp
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
#include <thread>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE    1000000000 //1e9
#define THREADS std::thread::hardware_concurrency()

void replace(int start, int end, int* array, int x, int y) {
    for (int i = start; i < end; i++) {
        if (array[i] == x) {
            array[i] = y;
        }
    }
}

int main(int argc, char* argv[]) {
    int *array, *aux;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point startTime, endTime;
    double timeElapsed;

    int start, end, remainder, block_size;
    thread threads[THREADS];
    
    block_size = SIZE / THREADS;
    remainder = SIZE % THREADS;

    array = new int[SIZE];
    for (int i = 0; i < SIZE; i++) {
        array[i] = 1;
    }
    display_array("before", array);
    
    aux = new int[SIZE];

    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        memcpy(aux, array, sizeof(int) * SIZE);
        
        startTime = high_resolution_clock::now();

        start = 0;
        for (int i = 0; i < THREADS; i++) {
            end = start + block_size + ((i < remainder)? 1 : 0);
            threads[i] = thread(replace, start, end, aux, 1, -1);
            start = end;
        }

        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
        }

        endTime = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(endTime - startTime).count();
    }
    
    display_array("after", aux);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] array;
    delete [] aux;
    
    return 0;
}
