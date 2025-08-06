// =================================================================
//
// File: intro01.cpp
// Author: Pedro Perez
// Description: This file implements a race condition problem.
//              To compile: g++ -o app -pthread intro01.cpp  
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <thread>

using namespace std;

#define THREADS     4
#define ITERATIONS  5

int counter = 0;

void increment(int id) {
    int prev;

    for (int i = 0; i < ITERATIONS; i++) {
        prev = counter++;
        cout << "id=" << id << ", previous = " 
             << prev << " current = " << counter << "\n";
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
