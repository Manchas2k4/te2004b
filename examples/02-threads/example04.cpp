// =================================================================
//
// File: example04.cpp
// Author: Pedro Perez
// Description: This file implements the algorithm to find the 
//		minimum value in an array using C/C++ threads. To 
//		compile:
//		g++ -o app -pthread example04.cpp
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
#include <cstring>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 	1000000000 // 1e9
#define THREADS std::thread::hardware_concurrency()

void minimum(int start, int end, int *array, int &result) {
    int local = array[start];
    for (int i = start + 1; i < end; i++) {
        if (array[i] < local) {
            local = array[i];
        }
    }
    result = local;
}

int main(int argc, char* argv[]) {
    int *array, result;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    int blockSize;
    int results[THREADS];
    thread threads[THREADS];

    array = new int [SIZE];
    
    random_array(array, SIZE);
    display_array("array:", array);

    memset(results, 0, sizeof(int) * THREADS);

    blockSize = ceil((double) SIZE / THREADS);
    
    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        start = high_resolution_clock::now();

        for (int i = 0; i < THREADS; i++) {
            int start = (i * blockSize);
            int end = (i != (THREADS - 1))? ((i + 1) * blockSize) : SIZE;
            threads[i] = thread(minimum, start, end, array, std::ref(results[i]));
        }

        result = array[0];
        for (int i = 0; i < THREADS; i++) {
            threads[i].join();
            if (results[i] < result) {
                result = results[i];
            }
        }

        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    cout << "result = " << result << "\n";
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    delete [] array;

    return 0;
}
