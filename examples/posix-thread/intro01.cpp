// =================================================================
//
// File: intro01.cpp
// Author: Pedro Perez
// Description: This file contains an example of POSIX thread 
//              handling in C/C++.
//              To compile: g++ intro01.cpp -lpthread
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

void* task(void *param) {
    int i;

    for (i = 0; i < LIMIT; i++) {
        cout << "PID = " << getpid() << " TID = " << pthread_self()
            << " i = " << i << "\n";
        sleep(1);
    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
  pthread_t tid;

  pthread_create(&tid, NULL, task, NULL);

  pthread_join(tid, NULL);

  return 0;
}
