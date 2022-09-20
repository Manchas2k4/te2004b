// =================================================================
//
// File: race-condition.cpp
// Author: Pedro Perez
// Description: This file implements a race condition problem.
//              To compile: g++ race-condition.cpp -lpthread
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

int counter = 0;

const int MAX_SLEEP_TIME = 3;
const int MAX_THREADS = 4;
const int MAX_ITERATIONS = 5;

void* increment(void *param) {
    int id, sleepTime;

    id = *((int*) param);
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        //cout << "id = " << id << " counter initial value = " << counter << "\n";
        counter++;
        sleepTime = (rand() % MAX_SLEEP_TIME) + 1;
        sleep(sleepTime);
        cout << "id = " << id << " incrementing, counter = " << counter << "\n";
    }
    pthread_exit(NULL);
}

void* decrement(void *param) {
    int id, sleepTime;

    id = *((int*) param);
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        counter--;
        sleepTime = (rand() % MAX_SLEEP_TIME) + 1;
        sleep(sleepTime);
        cout << "id = " << id << " decrementing, counter = " << counter << "\n";

    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    pthread_t tids[MAX_THREADS];
    int id[MAX_THREADS];

    for (int i = 0; i < MAX_THREADS; i++) {
        id[i] = i;
    }

    for (int i = 0; i < MAX_THREADS; i++) {
        if (i % 2 == 0) {
            pthread_create(&tids[i], NULL, increment, (void*) &id[i]);
        } else {
            pthread_create(&tids[i], NULL, decrement, (void*) &id[i]);
        }
    }

    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(tids[i], NULL);
    }

    return 0;
}
