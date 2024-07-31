// =================================================================
//
// File: intro04.cpp
// Author: Pedro Perez
// Description: This file implements a strategy that allows tasks to 
//				be alternated between two types of threads: type 1 
//				tasks are done first, while type 2 tasks wait and then 
//				they alternate.
//
//              To compile: g++ intro04.cpp -pthread -o app
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

#define TIMES	10

pthread_mutex_t type1_turn = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t type2_turn = PTHREAD_MUTEX_INITIALIZER;

void* type1(void *arg) {
	cout << "Type 1 starting...\n";
	int i = TIMES;
	while (i > 0) {
		pthread_mutex_lock(&type1_turn);
		cout << "Type 1 - " << i << "\n";
		pthread_mutex_unlock(&type2_turn);
		i--;
	}
	cout << "Type 1 ending\n";
	pthread_exit(NULL);
}

void* type2(void *arg) {
	cout << "Type 2 starting...\n";
	int i = 1;
	while (i <= TIMES) {
		pthread_mutex_lock(&type2_turn);
		cout << "Type 2 - " << i << "\n";
		pthread_mutex_unlock(&type1_turn);
		i++;
	}
	cout << "Type 2 ending\n";
	pthread_exit(NULL);
}


int main(int argc, char* argv[])   {
	pthread_t type1_thread;
	pthread_t type2_thread;

	pthread_mutex_lock(&type1_turn);

	pthread_create(&type1_thread, NULL, type1, NULL);
	std::this_thread::sleep_for(std::chrono::milliseconds(3000));

	pthread_create(&type2_thread, NULL, type2, NULL);
	
	pthread_join(type1_thread, NULL);
	pthread_join(type2_thread, NULL);
	return 0;
}