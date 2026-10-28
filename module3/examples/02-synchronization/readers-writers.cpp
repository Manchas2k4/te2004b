// =================================================================
//
// File: readers-writers.cpp
// Author: Pedro Perez
// Description: This file implements the readers-writers
//              synchronization problem using pthreads.
//              To compile: g++ readers-writers.cpp -lpthread -o app
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <cstdlib>

using namespace std;

const int MAX_READERS = 15;
const int MAX_WRITERS = 5;
const int MAX_TIMES = 5;
const int MAX_SLEEP_TIME = 5;

int readers = 0, writers = 0, waitingReaders = 0, waitingWriters = 0;
mutex condLock;
condition_variable canRead, canWrite;

/************* CODE FOR READERS *************/
void beginReading(int id) {
    unique_lock<std::mutex> lock(condLock);
    cout << "Reader " << id << " is waiting to read\n";
    while (writers == 1 || waitingWriters > 0) {
        waitingReaders++;
        canRead.wait(lock);
        waitingReaders--;
    }

    /****** CRITICAL SECTION ******/
    readers++;
    cout << "Reader " << id << " is now reading\n";
}

void endReading(int id) {
    unique_lock<std::mutex> lock(condLock);
    readers--;
    if (readers == 0) {
        canWrite.notify_one();
    }
    cout << "Reader " << id << " has finished reading\n";
}

void reader(int id) {
    for (int i = 0; i < MAX_TIMES; i++) {
        beginReading(id);
        this_thread::sleep_for(std::chrono::seconds(1));
        endReading(id);
        this_thread::sleep_for(std::chrono::seconds(rand() % MAX_SLEEP_TIME + 1));
    }
}
/************* CODE FOR READERS *************/

/************* CODE FOR WRITERS *************/
void beginWriting(int id) {
    unique_lock<std::mutex> lock(condLock);
    cout << "Writer " << id << " is waiting to write\n";
    while (writers == 1 || readers > 0) {
        waitingWriters++;
        canWrite.wait(lock);
        waitingWriters--;
    }

    /****** CRITICAL SECTION ******/
    writers = 1;
    cout << "Writer " << id << " is writing\n";
}

void endWriting(int id) {
    unique_lock<std::mutex> lock(condLock);
    writers = 0;
    if (waitingReaders > 0) {
        canRead.notify_all();
    } else {
        canWrite.notify_one();
    }
    cout << "Writer " << id << " has finished writing\n";
}

void writer(int id) {
    for (int i = 0; i < MAX_TIMES; i++) {
        beginWriting(id);
        this_thread::sleep_for(std::chrono::seconds(3));
        endWriting(id);
        this_thread::sleep_for(std::chrono::seconds(rand() % MAX_SLEEP_TIME + 1));
    }
}
/************* CODE FOR WRITERS *************/

int main() {
    srand(time(NULL));
    thread threads[MAX_READERS + MAX_WRITERS];
    int ids[MAX_READERS + MAX_WRITERS];

    for (int i = 0; i < MAX_READERS + MAX_WRITERS; i++) {
        ids[i] = i;
    }

    int j = 0;
    for (int i = 0; i < MAX_READERS; i++) {
        threads[j] = thread(reader, ids[j]);
        j++;
    }

    for (int i = 0; i < MAX_WRITERS; i++) {
        threads[j] = thread(writer, ids[j]);
        j++;
    }

    for (int i = 0; i < MAX_READERS + MAX_WRITERS; i++) {
        threads[i].join();
    }

    return 0;
}
