// =================================================================
//
// File: intro03.cpp
// Author: Pedro Perez
// Description: This file contains an example of POSIX thread 
//              handling in C/C++.
//              To compile: g++ intro03.cpp -lpthread
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

const int THREADS = 4;
const int LIMIT = 5;

using namespace std;

void* task(void* param) {
    int i, id;

    id = *((int*) param);
    for (i = 0; i < LIMIT; i++) {
        cout << "PID = " << getpid() << " ID = " << id
            << " i = " << i << "\n";
    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    pthread_t tid[THREADS];
    int i;

    for (i = 0; i < THREADS; i++) {
        pthread_create(&tid[i], NULL, task, (void*) &i);
    }

    for (i = 0; i < THREADS; i++) {
        pthread_join(tid[i], NULL);
    }

    return 0;
}
