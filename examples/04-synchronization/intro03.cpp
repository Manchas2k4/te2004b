// =================================================================
//
// File: intro03.cpp
// Author: Pedro Perez
// Description: This file implements a synchronization strategy on a 
//              shared variable using pthreads. Unlike the previous 
//              example (intro02.cpp), here the increment 
//              and decrement threads alternate.
//
//              To compile: g++ intro03.cpp -lpthread -o app
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================
#include <iostream>
#include <iomanip>
#include <pthread.h>

using namespace std;

int counter = 0;

const int MAX_THREADS = 4;
const int MAX_ITERATIONS = 5;

pthread_mutex_t add_lock, sub_lock;

void* increment(void *param) {
    int id, prev;

    id = *((int*) param);
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        pthread_mutex_lock(&add_lock);
        prev = counter++;
        cout << "incrementing id = " << id << ", previous = " 
             << prev << " current = " << counter << "\n";
        pthread_mutex_unlock(&sub_lock);
    }
    pthread_exit(NULL);
}

void* decrement(void *param) {
    int id, prev;

    id = *((int*) param);
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        pthread_mutex_lock(&sub_lock);
        prev = counter--;
        cout << "decrementing id = " << id << ", previous = " 
             << prev << " current = " << counter << "\n";
        pthread_mutex_unlock(&add_lock);
    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    pthread_t tids[MAX_THREADS];
    int id[MAX_THREADS];

    for (int i = 0; i < MAX_THREADS; i++) {
        id[i] = i;
    }
    pthread_mutex_init(&add_lock, NULL);
    pthread_mutex_init(&sub_lock, NULL);
    pthread_mutex_lock(&sub_lock);

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
