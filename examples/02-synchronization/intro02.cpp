// =================================================================
//
// File: intro02.cpp
// Author: Pedro Perez
// Description: This file implements a race condition problem.
//
//              To compile: g++ intro02.cpp -lpthread -o app
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

int counter = 0;

const int MAX_THREADS = 4;
const int MAX_ITERATIONS = 5;

void increment(int id) {
    int prev;

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        prev = counter++;
        cout << "id=" << id << ", previous = " 
             << prev << " current = " << counter << "\n";
    }
}

void decrement(int id) {
    int prev;

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        prev = counter--;
        cout << "decrementing id = " << id << ", previous = " 
             << prev << " current = " << counter << "\n";
    }
}

int main(int argc, char* argv[]) {
    thread tids[MAX_THREADS];

    for (int i = 0; i < MAX_THREADS; i++) {
        if (i % 2 == 0) {
            tids[i] = thread(increment, i);
        } else {
            tids[i] = thread(decrement, i);
        }
    }

    for (int i = 0; i < MAX_THREADS; i++) {
        tids[i].join();
    }

    return 0;
}
