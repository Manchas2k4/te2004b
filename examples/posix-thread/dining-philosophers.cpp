// =================================================================
//
// File: dinning-philosophers.cpp
// Author: Pedro Perez
// Description: This file implements the dining philosophers
//              synchronization problem using pthreads.
//              To compile: g++ dinning-philosophers.cpp -lpthread
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>

using namespace std;

const int MAX_PHILOSOPHERS = 5;
const int MAX_ITERATIONS = 5;
const int MAX_SLEEP_TIME = 5;
enum {THINKING, HUNGRY, EATING} state[MAX_PHILOSOPHERS];

pthread_t tids[MAX_PHILOSOPHERS];
int id[MAX_PHILOSOPHERS];
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
        pthread_cond_signal(&chopsticks[i]);
    }
}

void pickupChopsticks(int i) {
    pthread_mutex_lock(&mutex_lock);
    state[i] = HUNGRY;
    test(i);
    while (state[i] != EATING) {
        pthread_cond_wait(&chopsticks[i], &mutex_lock);
    }
    pthread_mutex_unlock(&mutex_lock);
}

void returnChopsticks(int i) {
    pthread_mutex_lock(&mutex_lock);
    state[i] = THINKING;
    test(left(i));
    test(right(i));
    pthread_mutex_unlock(&mutex_lock);
}

void thinking(int sleepTime) {
    sleep(sleepTime);
}

void eating(int sleepTime) {
    sleep(sleepTime);
}

void* philosopher(void *param) {
    int id = *((int*) param);
    int sleepTime;
    int i;

    srand(time(0) + id);
    while(i < MAX_ITERATIONS) {
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
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    for (int i = 0; i < MAX_PHILOSOPHERS; i++) {
        state[i] = THINKING;
        id[i] = i;
        pthread_cond_init(&chopsticks[i], NULL);
    }
    pthread_mutex_init(&mutex_lock, NULL);

    for (int i = 0; i < MAX_PHILOSOPHERS; i++) {
        pthread_create(&tids[i], NULL, philosopher, (void*) &id[i]);
    }

    for (int i = 0; i < MAX_PHILOSOPHERS; i++) {
        pthread_join(tids[i], NULL);
    }

    return 0;
}
