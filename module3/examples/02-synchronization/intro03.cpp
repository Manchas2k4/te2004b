// =================================================================
//
// File: intro03.cpp
// Author: Pedro Perez
// Description: This file implements the solution to solve a race 
//              condition problem.
//
//              To compile: g++ intro03.cpp -pthread -o app
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>

using namespace std;

const int MAX_THREADS = 4;
const int MAX_ITERATIONS = 5;

int counter = 0;
mutex m;


void increment(int id) {
    int prev;

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        m.lock();
        prev = counter++;
        cout << "id=" << id << ", previous = " 
             << prev << " current = " << counter << "\n";
        m.unlock();
    }
}

int main(int argc, char* argv[]) {
    thread tids[MAX_THREADS];
    int id[MAX_THREADS];

    for (int i = 0; i < MAX_THREADS; i++) {
        tids[i] = thread(increment, i);
    }

    for (int i = 0; i < MAX_THREADS; i++) {
        tids[i].join();
    }

    return 0;
}
