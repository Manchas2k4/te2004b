// =================================================================
//
// File: smokers.cpp
// Author: Pedro Perez
// Description: This file implements the smokers synchronization
//              problem using pthreads.
//              To compile: g++ readers-writers.cpp -lpthread -o app
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <thread>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <mutex>
#include <condition_variable>

using namespace std;

const int MAX_TIMES = 20;
const int TOBACCO = 0;
const int PAPER = 1;
const int MATCH = 2;

mutex tableLock, tobaccoLock, paperLock, matchLock;
condition_variable tobaccoCV, paperCV, matchCV;

void acquire(int resource) {
    switch (resource) {
        case TOBACCO : tobaccoLock.lock(); break;
        case PAPER   : paperLock.lock(); break;
        default      : matchLock.lock(); break;
    }
}

void release(int resource) {
    switch (resource) {
        case TOBACCO : 
            tobaccoLock.unlock();
            tobaccoCV.notify_one();
            break;
        case PAPER   : 
            paperLock.unlock();
            paperCV.notify_one();
            break;
        default      : 
            matchLock.unlock();
            matchCV.notify_one();
            break;
    }
}

string translate(int resource) {
    switch (resource) {
        case TOBACCO : return "tobacco";
        case PAPER   : return "paper";
        default      : return "match";
    }
}

void smoker(int resource) {
    cout << "The smoker with " << translate(resource)
         << " has started... waiting for other ingredients\n";
    while (true) {
        acquire(resource);
        cout << "The smoker with " << translate(resource)
             << " takes what the agent left, makes a cigar and smokes it..\n";
        this_thread::sleep_for(std::chrono::milliseconds(1000));
        tableLock.unlock();
    }
}

void agent() {
    for (int i = 0; i < MAX_TIMES; i++) {
    std:unique_lock<std::mutex> lock(tableLock);
    int value = (rand() % 3);
    switch (value) {
      case TOBACCO:
        cout << "Agent is placing paper and match.\n";
        release(TOBACCO);
        break;
      case PAPER:
        cout << "Agent is placing a tobacco and match.\n";
        release(PAPER);
        break;
      default:
        cout << "Agent is placing a tobacco and paper.\n";
        release(MATCH);
        break;
    }
  }
}

int main() {
    thread smokers[3], agentThread;
    int resources[] = {TOBACCO, PAPER, MATCH};

    tableLock.lock();
    tobaccoLock.lock();
    paperLock.lock();
    matchLock.lock();

    srand(time(0));

    for (int i = 0; i < 3; i++) {
        smokers[i] = std::thread(smoker, resources[i]);
    }

    agentThread = std::thread(agent);

    for (auto& smoker : smokers) {
        smoker.join();
    }
    agentThread.join();

    return 0;
}