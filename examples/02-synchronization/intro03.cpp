// =================================================================
//
// File: intro03.cpp
// Author: Pedro Perez
// Description: This file implements a synchronization strategy on a 
//              shared variable using threads. Unlike the previous 
//              example (intro02.cpp), here the increment 
//              and decrement threads alternate.
//              To compile: g++ -o app -pthread intro03.cpp 
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

const int THREADS = 4;
const int ITERATIONS = 5;

mutex add_lock, sub_lock;

void increment(int id) {
    int prev;

    for (int i = 0; i < ITERATIONS; i++) {
        add_lock.lock();
        prev = counter++;
        cout << "incrementing id = " << id << ", previous = " 
             << prev << " current = " << counter << "\n";
        sub_lock.unlock();
    }
}

void decrement(int id) {
    int prev;

    for (int i = 0; i < ITERATIONS; i++) {
        sub_lock.lock();
        prev = counter--;
        cout << "decrementing id = " << id << ", previous = " 
             << prev << " current = " << counter << "\n";
        add_lock.unlock();
    }
}

int main(int argc, char* argv[]) {
    thread threads[THREADS];
    
    sub_lock.lock();

    for (int i = 0; i < THREADS; i++) {
        if (i % 2 == 0) {
            threads[i] = thread(increment, i);
        } else {
            threads[i] = thread(decrement, i);
        }
    }

    for (int i = 0; i < THREADS; i++) {
        threads[i].join();
    }

    return 0;
}
