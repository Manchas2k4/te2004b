// =================================================================
//
// File: intro06.cpp
// Author: Pedro Perez
// Description: This file implements a synchronization strategy where
//              a group of threads must wait for all of them to reach 
//              a common point before they can continue their 
//              execution. REUSABLE BARRIER.
//
//              To compile: g++ intro06.cpp -pthread -o app
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================
#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;

const int MAX_THREADS = 4;
const int MAX_ITERATIONS = 3;

int counter = 0;
int steps = 0;
mutex semaphore, display;
condition_variable cv;

void task(int id) {
    int sleep = ((rand() % 5) + id);

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        display.lock();
        cout << "id=" << id << " before the barrier.\n";
        cout << "Thread " << id << " going to sleep for " << sleep << " seconds\n\n";
        display.unlock();

	    std::this_thread::sleep_for(std::chrono::seconds(sleep));


        unique_lock<mutex> barrier(semaphore);
        int current_step = steps;
        counter++;
        if (counter == MAX_THREADS) {
            counter = 0;
            steps++;
            cv.notify_all();
        } else {
            cv.wait(barrier, [current_step] { return current_step != steps; });
        }

        cout << "\t\tid=" << id << " step=" << i <<  " after the barrier.\n";
    }
}

int main(int argc, char* argv[]) {
    thread tids[MAX_THREADS];

    for (int i = 0; i < MAX_THREADS; i++) {
        tids[i] = thread(task, i);
    }

    for (int i = 0; i < MAX_THREADS; i++) {
        tids[i].join();
    }

    return 0;
}
