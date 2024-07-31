// =================================================================
//
// File: intro05.cpp
// Author: Pedro Perez
// Description: This file implements a strategy that allows tasks to 
//				be alternated between two types of threads: type 1 
//				tasks are done first, while type 2 tasks wait and then 
//				they alternate.
//
//              To compile: g++ intro05.cpp -pthread -o app
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <pthread.h>
#include <thread>

using namespace std;

#define THREADS	12

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t barrier = PTHREAD_MUTEX_INITIALIZER;

int count = 0;

void* task(void *param) {
	int id = *((int*) param);
	int sleep = ((rand() % 3000) + 1);

	cout << "Thread " << id << " going to sleep for " << sleep << " miliseconds\n";
	std::this_thread::sleep_for(std::chrono::milliseconds(sleep));

	pthread_mutex_lock(&mutex);
	count++;
	cout << "Thread " << id << " count = " << count << "\n";
	pthread_mutex_unlock(&mutex);

	if (count == THREADS) {
		pthread_mutex_unlock(&barrier);
	}

	pthread_mutex_lock(&barrier);
	pthread_mutex_unlock(&barrier);
	cout << "Thread " << id << " critical point\n";

	return 0;
}

int main(int argc, char* argv[])   {
	pthread_t tids[THREADS];

	pthread_mutex_lock(&barrier);

	for (int i = 0; i < THREADS; i++) {
		pthread_create(&tids[i], NULL, task, &i);
	}

	for (int i = 0; i < THREADS; i++) {
		pthread_join(tids[i], NULL);
	}

	return 0;
}