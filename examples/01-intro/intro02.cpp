// =================================================================
//
// File: intro02.cpp
// Author: Pedro Perez
// Description: This file show how to send paramenters to a 
//              simple thread. To compile:
//				g++ -o app -pthread intr02.cpp
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

void task (int limit) {
    for (int i = 1; i <= limit; i++) {
        cout << i << " ";
    }
    cout << "\n";
}

int main(int argc, char* argv[]) {
    thread t;

    t = thread(task, 20);

    t.join();

    return 0;
}
