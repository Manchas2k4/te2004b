// =================================================================
//
// File: intro01.cpp
// Author: Pedro Perez
// Description: This file implements a race condition problem.
//
//              To compile: g++ intro01.cpp -pthread -o app
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

const int MAX_THREADS = 4;
const int MAX_ITERATIONS = 5;

int counter = 0;

void* increment(void *param) {
    int id, prev;

    id = *((int*) param);
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        prev = counter++;
        cout << "id=" << id << ", previous = " 
             << prev << " current = " << counter << "\n";
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
        pthread_create(&tids[i], NULL, increment, (void*) &id[i]);
    }

    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(tids[i], NULL);
    }

    return 0;
}
