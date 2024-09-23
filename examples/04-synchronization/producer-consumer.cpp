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
#include <queue>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>

using namespace std;
using namespace std::chrono;

#define SIZE 	10
#define N		5

mutex mtx;
condition_variable not_empty;
condition_variable not_full;
queue<int> buffer;
bool finished;

void producer(int id) {
    while(!finished) {
		unique_lock<mutex> lock(mtx);

		not_full.wait(lock, [] () { return buffer.size() != SIZE; });

		cout << "\tBuffer size before producing " << buffer.size() << "\n";
		buffer.push(id);
		cout << "\tProducer " << id << " adding " << (id * 10) << "\n";
		cout << "\tBuffer size after producing " << buffer.size() << "\n\n";

		lock.unlock();

		not_empty.notify_one();

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
}

void consumer(int id) {
	int value;

	while (!finished) {
		unique_lock<mutex> lock(mtx);

		not_empty.wait(lock, [] ()  { return buffer.size() != 0; });

		cout << "Buffer size before consuming " << buffer.size() << "\n";
		int val = buffer.front(); buffer.pop();
		cout << "Consumer " << id << " taking " << (id * 10) << "\n";
		cout << "Buffer size after consuming " << buffer.size() << "\n\n";

		lock.unlock();

		not_full.notify_one();

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
}

int main(int argc, char* argv[])   {
	thread producer_thread[N];
    thread consumer_thread[N];

	finished = false;

	for (int i = 0; i < N; i++) {
        producer_thread[i] = thread(producer, (i + 1));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(5000));

    for (int i = 0; i < N; i++) {
        consumer_thread[i] = thread(consumer, (i + 1));
    }

	std::this_thread::sleep_for(std::chrono::milliseconds(5000));
	
	{
		lock_guard<std::mutex> lock(mtx);
		finished = true;
		cout << "Finish them!!\n";
	}

	for (int i = 0; i < N; i++) {
		producer_thread[i].join();
		consumer_thread[i].join();
	}

	return 0;
}
