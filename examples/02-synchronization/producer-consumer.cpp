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
#include <mutex>
#include <condition_variable>

using namespace std;

const int SIZE = 10;
const int MAXNUM = 10000;
const int MAXPROD = 5;
const int MAXCON = 5;

mutex mtx;
condition_variable space_available;
condition_variable data_available;

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
	int i;

	cout << "Producter " << i << " starting...\n";
	i = 0;
	while (1) {
		unique_lock<std::mutex> lock(mtx);
        space_available.wait(lock, [] { return count < SIZE; });

        cout << "Producer " << id << " adding " << i 
			 << " - Buffer elements = " << count << "\n";
        add_buffer(i);
        data_available.notify_one();

        lock.unlock();
        i = (i + 1) % MAXNUM;
        this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
	cout << "Producter " << i << " ending...\n";
}

void consumer(int id) {
    cout << "Consumer " << id << " starting...\n";
    for (int i = 0; i < 10; ++i) {
        unique_lock<std::mutex> lock(mtx);
        data_available.wait(lock, [] { return count > 0; });

        int v = get_buffer();
        cout << "Consumer " << id << " getting " << v 
			 << " - Buffer elements = " << count << "\n";
        space_available.notify_one();

        lock.unlock();
    }
    cout << "Consuming " << id << " ending...\n";
}

int main() {
    thread producer_threads[MAXPROD];
    thread consumer_threads[MAXCON];

    for (int i = 0; i < MAXPROD; ++i) {
        producer_threads[i] = thread(producer, i);
    }

    this_thread::sleep_for(std::chrono::milliseconds(10000));

    for (int i = 0; i < MAXCON; ++i) {
        consumer_threads[i] = thread(consumer, i);
    }
    
    for (int i = 0; i < MAXCON; ++i) {
        consumer_threads[i].join();
    }

    return 0;
}