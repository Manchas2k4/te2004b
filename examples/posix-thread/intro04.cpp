// =================================================================
//
// File: intro04.cpp
// Author: Pedro Perez
// Description: This file contains an example of POSIX thread 
//              handling in C/C++.
//              To compile: g++ intro04.cpp -lpthread
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

// limite [inf, sup)
typedef struct {
  int id, inf, sup;
} Limits;

void* task(void* param) {
    int i;
    Limits *limits;

    limits = (Limits*) param;
    for (i = limits->inf; i < limits->sup; i++) {
        cout << "PID = " << getpid() << " ID = " << limits->id
            << " i = " << i << "\n";
    }
    free(limits);
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    pthread_t tid[THREADS];
    Limits *limits;
    int i;

    for (i = 0; i < THREADS; i++) {
        limits = (Limits*) malloc(sizeof(Limits));
        limits->id = i;
        limits->inf = i * 100;
        limits->sup = (i + 1) * 100;
        pthread_create(&tid[i], NULL, task, (void*) limits);
    }

    for (i = 0; i < THREADS; i++) {
        pthread_join(tid[i], NULL);
    }

    return 0;
}
