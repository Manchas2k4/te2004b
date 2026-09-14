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
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

const int MAX_PHILOSOPHERS = 5;
const int MAX_ITERATIONS = 5;
const int MAX_SLEEP_TIME = 5000;
enum {THINKING, HUNGRY, EATING} state[MAX_PHILOSOPHERS];

mutex mutex_lock;
condition_variable chopsticks[MAX_PHILOSOPHERS];

int left(int i) {
    return (i == 0) ? (MAX_PHILOSOPHERS - 1) : (i - 1);
}

int right(int i) {
    return (i == MAX_PHILOSOPHERS - 1) ? 0 : (i + 1);
}

void test(int i) {
    if (state[i] == HUNGRY && 
        state[left(i)] != EATING && 
        state[right(i)] != EATING) {
        state[i] = EATING;
        chopsticks[i].notify_one();
    }
}

void pickupChopsticks(int i) {
    unique_lock<std::mutex> lock(mutex_lock);
    state[i] = HUNGRY;
    test(i);
    while (state[i] != EATING) {
        chopsticks[i].wait(lock);
    }
}

void returnChopsticks(int i) {
    unique_lock<std::mutex> lock(mutex_lock);
    state[i] = THINKING;
    test(left(i));
    test(right(i));
}

void thinking(int sleepTime) {
    this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
}

void eating(int sleepTime) {
    this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
}

void philosopher(int id) {
    int sleepTime;
    int i = 0;

    srand(time(0) + id);
    cout << "Philosopher " << id << " is starting...\n";
    while (i < MAX_ITERATIONS) {
        sleepTime = (rand() % MAX_SLEEP_TIME) + 1;
        thinking(sleepTime);

        pickupChopsticks(id);
        cout << "Philosopher " << id << " is eating\n";
        sleepTime = (rand() % MAX_SLEEP_TIME) + 1;
        eating(sleepTime);
        returnChopsticks(id);

        cout << "Philosopher " << id << " is thinking\n";
        i++;
    }
}

int main() {
    vector<thread> threads(MAX_PHILOSOPHERS);

    for (int i = 0; i < MAX_PHILOSOPHERS; i++) {
        state[i] = THINKING;
    }

    for (int i = 0; i < MAX_PHILOSOPHERS; i++) {
        threads[i] = std::thread(philosopher, i);
    }

    for (auto& th : threads) {
        th.join();
    }

    return 0;
}