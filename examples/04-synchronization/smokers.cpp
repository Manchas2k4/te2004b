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
#include <random>
#include <thread>
#include <mutex>

using namespace std;

#define TOBACCO     0
#define PAPER       1
#define MATCH       2

bool finished;
default_random_engine generator;
uniform_int_distribution<int> distribution(0, 2);
mutex mtx, tableLock, tobaccoLock, paperLock, matchLock;

void acquire(int resource) {
  switch (resource) {
    case TOBACCO : tobaccoLock.lock(); break;
    case PAPER   : paperLock.lock(); break;
    default      : matchLock.lock(); break;
  }
}

void release(int resource) {
  switch (resource) {
    case TOBACCO : tobaccoLock.unlock(); break;
    case PAPER   : paperLock.unlock(); break;
    default      : matchLock.unlock(); break;
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
  while(!finished) {
    acquire(resource);
    cout << "The smoker with " << translate(resource)
         << " take what the agent left, makes a cigar and smokes it..\n";
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		tableLock.unlock();
  }
  cout << "The smoker with " << translate(resource)
       << " has finished\n";
}

void agent() {
  int value;

  cout << "The Agent is starting...\n";
  while (!finished) {
    tableLock.lock();
    value = distribution(generator);
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

  finished = false;

  tobaccoLock.lock();
  paperLock.lock();
  matchLock.lock();

  srand(time(0));
  for (int i = 0; i < 3; i++) {
    smokers[i] = thread(smoker, resources[i]);
  }

  agents = thread(agent);

  std::this_thread::sleep_for(std::chrono::milliseconds(10000));

  {
		lock_guard<std::mutex> lock(mtx);
		finished = true;
		cout << "Finish them!!\n";
	}

  for (int i = 0; i < 3; i++) {
    smokers[i].join();
  }
  agents.join();

  return 0;
}
