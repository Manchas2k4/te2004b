// =================================================================
//
// File: producer-consumer.cpp
// Author: Pedro Perez
// Description: This file implements the producer-consumer
//              synchronization problem using pthreads.
//              To compile: g++ producer-consumer.cpp -lpthread -o app
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

const int SIZE = 10;
const int MAXNUM = 10000;
const int MAXPROD = 5;
const int MAXCON = 5;

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

void* producer(void *arg) {
	int i;

	printf("producter starting...\n");
	i = 0;
	while (1) {
		pthread_mutex_lock(&mutex);
		if (count == SIZE) {
			pthread_cond_wait(&space_available, &mutex);
		}
		printf("producer adding %i...\n", i);
		add_buffer(i);
		pthread_cond_signal(&data_available);
		pthread_mutex_unlock(&mutex);
		i = (i + 1) % MAXNUM;
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
	pthread_exit(NULL);
}

void* consumer(void *arg) {
	int v;
	printf("consumer starting...\n");
	for (int i = 0; i < 10; i++) {
		pthread_mutex_lock(&mutex);
		if (count == 0) {
			pthread_cond_wait(&data_available, &mutex);
		}
		v = get_buffer();
		printf("consumer getting %i...\n", v);
		pthread_cond_signal(&space_available);
		pthread_mutex_unlock(&mutex);
	}
	printf("consuming finishing...\n");
	pthread_exit(NULL);
}

int main(int argc, char* argv[])   {
	pthread_t producer_thread[MAXPROD];
	pthread_t consumer_thread[MAXCON];

	for (int i = 0; i < MAXPROD; i++) {
		pthread_create(&producer_thread[i], NULL, producer, NULL);
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(10000));
	for (int i = 0; i < MAXCON; i++) {
		pthread_create(&consumer_thread[i], NULL, consumer, NULL);
	}
	for (int i = 0; i < MAXCON; i++) {
		pthread_join(consumer_thread[i], NULL);
	}
	return 0;
}
