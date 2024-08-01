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
#include <pthread.h>
#include <thread>

using namespace std;

#define MAX_PHILOSOPHERS    5
#define MAX_ITERATIONS      5
#define MAX_SLEEP_TIME      2000
enum {THINKING, HUNGRY, EATING} state[MAX_PHILOSOPHERS];

thread threads[MAX_PHILOSOPHERS];
pthread_cond_t chopsticks[MAX_PHILOSOPHERS];
pthread_mutex_t mutex_lock;

int left(int i) {
    return (i == 0)? (MAX_PHILOSOPHERS - 1) : (i - 1);
}

int right(int i) {
    return (i == MAX_PHILOSOPHERS - 1)? 0 : (i + 1);
}

void test(int i) {
    if (state[i] == HUNGRY &&
        state[left(i)] != EATING &&
        state[right(i)] != EATING) {

        state[i] = EATING;

        cout << "Philosopher " << i << " takes fork " << left(i) 
             << " and " << right(i) << std::endl;
        cout << "Philosopher " << i << " is eating\n";
        pthread_cond_signal(&chopsticks[i]);
    }
}

void pickupChopsticks(int i) {
    pthread_mutex_lock(&mutex_lock);
    state[i] = HUNGRY;
    cout << "Philosopher " << i << " is hungry" << std::endl;
    test(i);
    while (state[i] != EATING) {
        pthread_cond_wait(&chopsticks[i], &mutex_lock);
    }
    pthread_mutex_unlock(&mutex_lock);
}

void returnChopsticks(int i) {
    pthread_mutex_lock(&mutex_lock);
    state[i] = THINKING;
    cout << "Philosopher " << i << " putting fork " << left(i) 
         << " and " << right(i) << " down" << std::endl;
    cout << "Philosopher " << i << " is thinking" << std::endl;
    test(left(i));
    test(right(i));
    pthread_mutex_unlock(&mutex_lock);
}

void thinking(int sleepTime) {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
}

void eating(int sleepTime) {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
}

void philosopher(int id) {
    int sleepTime;

    cout << "Philosopher " << id << " is starting...\n";
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        sleepTime = (rand() % MAX_SLEEP_TIME) + 1;
        thinking(sleepTime);

        pickupChopsticks(id);
        
        sleepTime = (rand() % MAX_SLEEP_TIME) + 1;
        eating(sleepTime);

        returnChopsticks(id);

        i++;
    }
    cout << "Philosopher " << id << " is ending.\n";
}

int main(int argc, char* argv[]) {
    for (int i = 0; i < MAX_PHILOSOPHERS; i++) {
        state[i] = THINKING;
        pthread_cond_init(&chopsticks[i], NULL);
    }
    pthread_mutex_init(&mutex_lock, NULL);

    for (int i = 0; i < MAX_PHILOSOPHERS; i++) {
        threads[i] = thread(philosopher, i);
    }

    for (int i = 0; i < MAX_PHILOSOPHERS; i++) {
        threads[i].join();
    }

    return 0;
}
