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
#include <pthread.h>
#include <cstdlib>
#include <ctime>

using namespace std;

#define ITER_AGENT  15
#define ITER_SMOKE  5
#define TOBACCO     0
#define PAPER       1
#define MATCH       2
pthread_mutex_t tableLock, tobaccoLock, paperLock, matchLock;

void acquire(int resource) {
  switch (resource) {
    case TOBACCO : pthread_mutex_lock(&tobaccoLock); break;
    case PAPER   : pthread_mutex_lock(&paperLock); break;
    default      : pthread_mutex_lock(&matchLock); break;
  }
}

void release(int resource) {
  switch (resource) {
    case TOBACCO : pthread_mutex_unlock(&tobaccoLock); break;
    case PAPER   : pthread_mutex_unlock(&paperLock); break;
    default      : pthread_mutex_unlock(&matchLock); break;
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
  for (int i = 0; i < ITER_SMOKE; i++) {
    acquire(resource);
    cout << "The smoker with " << translate(resource)
         << " take what the agent left, makes a cigar and smokes it..\n";
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		pthread_mutex_unlock(&tableLock);
  }
  cout << "The smoker with " << translate(resource)
       << " has finished\n";
}

void agent() {
  int value;

  cout << "The Agent is starting...\n";
  for (int i = 0; i < ITER_AGENT; i++) {
    pthread_mutex_lock(&tableLock);
    value = (rand() % 3);
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
  cout << "The Agent has finished.\n";
}

int main(int argc, char* argv[]) {
  thread smokers[3], agents;
  int resources[] = {TOBACCO, PAPER, MATCH};

  pthread_mutex_init(&tableLock, NULL);
  pthread_mutex_init(&tobaccoLock, NULL);
  pthread_mutex_init(&paperLock, NULL);
  pthread_mutex_init(&matchLock, NULL);

  pthread_mutex_lock(&tobaccoLock);
  pthread_mutex_lock(&paperLock);
  pthread_mutex_lock(&matchLock);

  srand(time(0));
  for (int i = 0; i < 3; i++) {
    smokers[i] = thread(smoker, resources[i]);
  }

  agents = thread(agent);

  for (int i = 0; i < 3; i++) {
    smokers[i].join();
  }
  agents.join();

  return 0;
}
