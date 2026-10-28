// =================================================================
//
// File: intro04.cpp
// Author: Pedro Perez
// Description: This file implements a synchronization strategy on a 
//              shared variable using pthreads. Unlike the previous 
//              example (intro03.cpp), here the increment 
//              and decrement threads alternate.
//
//              To compile: g++ intro04.cpp -pthread -o app
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

int counter = 0;

const int MAX_THREADS = 4;
const int MAX_ITERATIONS = 5;

mutex add_lock, sub_lock;

void increment(int id) {
    int prev;

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        add_lock.lock();
        prev = counter++;
        cout << "incrementing id = " << id << ", previous = " 
             << prev << " current = " << counter << "\n";
        sub_lock.unlock();
    }
}

void decrement(int id) {
    int prev;

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        sub_lock.lock();
        prev = counter--;
        cout << "decrementing id = " << id << ", previous = " 
             << prev << " current = " << counter << "\n";
        add_lock.unlock();
    }
}

int main(int argc, char* argv[]) {
    thread tids[MAX_THREADS];

    add_lock.lock();

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
