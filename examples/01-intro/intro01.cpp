// =================================================================
//
// File: intro01.cpp
// Author: Pedro Perez
// Description: This file show how to create a simple thread.
//              To compile:
//              g++ -o app -pthread intr01.cpp
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

void task () {
    for (int i = 1; i <= 10; i++) {
        cout << i << " ";
    }
    cout << "\n";
}

int main(int argc, char* argv[]) {
    thread t;

    t = thread(task);

    t.join();

    return 0;
}
