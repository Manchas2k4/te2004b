//================================================================
//
// File: intro03.cpp
// Author: Pedro Perez
// Description: This file show how the problem with shared variable.
//              To compile:
//				g++ -o app -pthread intr03.cpp
//
// Copyright (c) 2023 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <thread>

using namespace std;

#define THREADS 4

void task (int id) {
    cout << "id = " << id << " has started.\n";
    for (int i = 1; i <= 20; i++) {
        cout << i << " ";
    }
    cout << "\n";
    cout << "id " << id << " has ended.\n";
}

int main(int argc, char* argv[]) {
    thread threads[THREADS];
    int limit = 20;

    for (int i = 0; i < THREADS; i++) {
        threads[i] = thread(task, i);
    }

    for (int i = 0; i < THREADS; i++) {
        threads[i].join();
    }

    return 0;
}
