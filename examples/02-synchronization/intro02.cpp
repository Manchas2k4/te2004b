// =================================================================
//
// File: intro02.cpp
// Author: Pedro Perez
// Description: This file implements the solution to solve a race 
//              condition problem.
//              To compile: g++ -o app -pthread intro02.cpp  
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

#define THREADS     4
#define ITERATIONS  5

int counter = 0;

mutex mutex_lock;

void increment(int id) {
    int prev;

    for (int i = 0; i < ITERATIONS; i++) {
        mutex_lock.lock();
        prev = counter++;
        cout << "id=" << id << ", previous = " 
             << prev << " current = " << counter << "\n";
        mutex_lock.unlock();
    }
}

int main(int argc, char* argv[]) {
    thread threads[THREADS];
    
    for (int i = 0; i < THREADS; i++) {
        threads[i] = thread(increment, i);
    }

    for (int i = 0; i < THREADS; i++) {
        threads[i].join();
    }

    return 0;
}
