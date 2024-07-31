// =================================================================
//
// File: intro04.cpp
// Author: Pedro Perez
// Description: This file show how to send a block of data to a 
//              thread. To compile:
//				g++ -o app -pthread intr04.cpp
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <thread>

using namespace std;

#define THREADS 4

typedef struct {
    int id, start, end;
} Block;

void task (Block & b) {
    cout << "id " << b.id << " has started.\n";
    for (int i = b.start; i < b.end; i++) {
        cout << i << " ";
    }
    cout << "\n";
    cout << "id " << b.id << " has ended.\n";
}

int main(int argc, char* argv[]) {
    thread threads[THREADS];
    Block blocks[THREADS];

    for (int i = 0; i < THREADS; i++) {
        blocks[i].id = i; 
        blocks[i].start = i * 100; 
        blocks[i].end = (i + 1) * 100; 
    }

    for (int i = 0; i < THREADS; i++) {
        threads[i] = thread(task, std::ref(blocks[i]));
    }

    for (int i = 0; i < THREADS; i++) {
        threads[i].join();
    }

    return 0;
}
