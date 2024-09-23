// =================================================================
//
// File: dinning-philosophers.cpp
// Author: Pedro Perez
// Description: This file implements the dining philosophers
//              synchronization problem using pthreads.
//              To compile: g++ dinning-philosophers.cpp -lpthread
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
#include <condition_variable>

using namespace std;

#define PHILOSOPHERS    5
#define SLEEP           1000

enum {THINKING, HUNGRY, EATING} state[PHILOSOPHERS];
mutex chopsticks[PHILOSOPHERS];
thread threads[PHILOSOPHERS];
mutex mtx;
condition_variable cond_var;
bool finished;

int left(int i) {
    return ((i + (PHILOSOPHERS - 1)) % PHILOSOPHERS);
}

int right(int i) {
    return ((i + 1) % PHILOSOPHERS);
}

bool can_eat(int i) {
    return (state[i] == HUNGRY &&
        state[left(i)] != EATING &&
        state[right(i)] != EATING);
}

void philosopher(int i) {
    cout << "Philosopher " << i << " is starting...\n";
    while (!finished) {
        unique_lock<std::mutex> lock(mtx);

        state[i] = HUNGRY;
        cout << "Philosopher " << i << " is hungry" << std::endl;
        cond_var.wait(lock, [i]() { return can_eat(i); });

        state[i] = EATING;
        cout << "Philosopher " << i << " takes fork " << left(i) 
                << " and " << right(i) << std::endl;

        lock.unlock();

        cout << "Philosopher " << i << " is eating\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        lock.lock();

        state[i] = THINKING;
        cout << "Philosopher " << i << " putting fork " << left(i) 
            << " and " << right(i) << " down" << std::endl;

        cout << "Philosopher " << i << " is thinking" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        cond_var.notify_all();
    }
    cout << "Philosopher " << i << " is ending.\n";
}

int main(int argc, char* argv[]) {
    finished = false;

    for (int i = 0; i < PHILOSOPHERS; i++) {
        state[i] = THINKING;
    }
    
    for (int i = 0; i < PHILOSOPHERS; i++) {
        threads[i] = thread(philosopher, i);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10000));

    {
		lock_guard<std::mutex> lock(mtx);
		finished = true;
		cout << "Finish them!!\n";
	}

	for (int i = 0; i < PHILOSOPHERS; i++) {
		threads[i].join();
	}

    return 0;
}
