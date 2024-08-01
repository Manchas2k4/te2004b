// =================================================================
//
// File: intro04.cpp
// Author: Pedro Perez
// Description: This file implements a barrier.
//              To compile: g++ -o app -pthread intro04.cpp 
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

using namespace std;

#define THREADS std::thread::hardware_concurrency()

mutex mutex_lock, barrier;

int count = 0;

void task(int id) {
	int sleep = ((rand() % 3000) + 1);

	cout << "Thread " << id << " going to sleep for " << sleep << " miliseconds\n";
	std::this_thread::sleep_for(std::chrono::milliseconds(sleep));

	mutex_lock.lock();
	count++;
	cout << "Thread " << id << " count = " << count << "\n";
	mutex_lock.unlock();

	if (count == THREADS) {
		barrier.unlock();
	}

	barrier.lock();
	barrier.unlock();
	cout << "Thread " << id << " critical point\n";
}

int main(int argc, char* argv[])   {
	thread threads[THREADS];

	barrier.lock();

	for (int i = 0; i < THREADS; i++) {
		threads[i] = thread(task, i);
	}

	for (int i = 0; i < THREADS; i++) {
		threads[i].join();
	}

	return 0;
}