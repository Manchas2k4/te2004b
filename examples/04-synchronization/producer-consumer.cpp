// =================================================================
//
// File: producer-consumer.cpp
// Author: Pedro Perez
// Description: This file implements the producer-consumer
//              synchronization problem using pthreads.
//              To compile: g++ -o app -pthread producer-consumer.cpp  
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <pthread.h>

using namespace std;
using namespace std::chrono;

#define SIZE 		10
#define MAXPROD 	5
#define MAXCON 		5
#define ITERATIONS 	10
#define SLEEPTIME	1000

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t space_available = PTHREAD_COND_INITIALIZER;
pthread_cond_t data_available = PTHREAD_COND_INITIALIZER;

int b[SIZE];
int count = 0;
int front = 0, rear = 0;

void add_buffer(int i) {
	b[rear] = i;
	rear = (rear + 1) % SIZE;
	count++;
}

int get_buffer(){
	int v;
	v = b[front];
	front = (front + 1) % SIZE;
	count--;
	return v ;
}

void producer(int id) {
	int value = id * 10;

    cout << "Producer " << id << " starting...\n";
	for (int i = 0; i < ITERATIONS; i++) {
		pthread_mutex_lock(&mutex);
		if (count == SIZE) {
			pthread_cond_wait(&space_available, &mutex);
		}
		cout << "Producter " << id << " adding " << i << "\n";
		add_buffer(i);
		pthread_cond_signal(&data_available);
		pthread_mutex_unlock(&mutex);
		std::this_thread::sleep_for(std::chrono::milliseconds(SLEEPTIME));
	}
	cout << "Producer " << id << " ending.\n";
}

void consumer(int id) {
	int value;

	cout << "Consumer " << id << " starting...\n";
    for (int i = 0; i < ITERATIONS; i++) {
		pthread_mutex_lock(&mutex);
		if (count == 0) {
			pthread_cond_wait(&data_available, &mutex);
		}
		value = get_buffer();
		cout << "Consumer " << id << " taking " << value << "\n";
		pthread_cond_signal(&space_available);
		pthread_mutex_unlock(&mutex);
	}
	cout << "Consumer " << id << " ending.\n";
}

int main(int argc, char* argv[])   {
	thread producer_thread[MAXPROD];
    thread consumer_thread[MAXCON];

	for (int i = 0; i < MAXPROD; i++) {
        producer_thread[i] = thread(producer, i);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(5000));

    for (int i = 0; i < MAXCON; i++) {
        consumer_thread[i] = thread(consumer, i);
    }

    for (int i = 0; i < MAXPROD; i++) {
        producer_thread[i].join();
    }

    for (int i = 0; i < MAXCON; i++) {
        consumer_thread[i].join();
    }
}
