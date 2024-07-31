// =================================================================
//
// File: readers-writers.cpp
// Author: Pedro Perez
// Description: This file implements the readers-writers
//              synchronization problem using pthreads.
//              To compile: g++ readers-writers.cpp -lpthread -o app
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>

using namespace std;

const int MAX_READERS = 15;
const int MAX_WRITERS = 5;
const int MAX_TIMES = 5;
const int MAX_SLEEP_TIME = 5;

int readers, writers, waitingReaders, waitingWriters;
pthread_cond_t canRead, canWrite;
pthread_mutex_t condLock;

/************* CODE FOR READERS *************/
void beginReading(int id) {
  pthread_mutex_lock(&condLock);
  cout << "Reader " << id << " is waiting to read\n";
  if (writers == 1 || waitingWriters > 0) {
    waitingReaders++;
    pthread_cond_wait(&canRead, &condLock);
    waitingReaders--;
  }

  /****** CRITICAL SECTION ******/
  readers++;
  cout << "Reader " << id << " is now reading\n";
  pthread_mutex_unlock(&condLock);
  pthread_cond_broadcast(&canRead);
  /****** CRITICAL SECTION ******/
}

void endReading(int id) {
  pthread_mutex_lock(&condLock);

  readers--;
  if (readers == 0) {
    pthread_cond_signal(&canWrite);
  }
  cout << "Reader " << id << " has finished reading\n";
  pthread_mutex_unlock(&condLock);
}

void* reader(void *param) {
  int id = *(int*) param;

  for (int i = 0; i < MAX_TIMES; i++) {
    beginReading(id);
    sleep(1);
    endReading(id);
    sleep((MAX_SLEEP_TIME % 5) + 1);
  }
  pthread_exit(0);
}
/************* CODE FOR READERS *************/
/************* CODE FOR WRITERS *************/
void beginWriting(int id) {
  pthread_mutex_lock(&condLock);
  cout << "Writer " << id << " is waiting to write\n";
  if (writers == 1 || readers > 0) {
    waitingWriters++;
    pthread_cond_wait(&canWrite, &condLock);
    waitingWriters--;
  }

  /****** CRITICAL SECTION ******/
  writers = 1;
  cout << "Writer " << id << " is writing\n";
  pthread_mutex_unlock(&condLock);
  /****** CRITICAL SECTION ******/
}

void endWriting(int id) {
  pthread_mutex_lock(&condLock);
  writers = 0;

  if (waitingReaders > 0) {
    pthread_cond_signal(&canRead);
  } else {
    pthread_cond_signal(&canWrite);
  }
  pthread_mutex_unlock(&condLock);
}

void* writer(void *param) {
  int id = *(int*) param;

  for (int i = 0; i < MAX_TIMES; i++) {
    beginWriting(id);
    sleep(3);
    endWriting(id);
    sleep((MAX_SLEEP_TIME % 5) + 1);
  }
  pthread_exit(0);
}
/************* CODE FOR WRITERS *************/

int main(int argc, char* argv[]) {
  int i, j;
  pthread_t tids[MAX_READERS + MAX_WRITERS];
  int ids[MAX_READERS + MAX_WRITERS];

  readers = writers = waitingReaders = waitingWriters = 0;
  pthread_cond_init(&canRead, NULL);
  pthread_cond_init(&canWrite, NULL);
  pthread_mutex_init(&condLock, NULL);

  for (i = 0; i < (MAX_READERS + MAX_WRITERS); i++) {
    ids[i] = i;
  }

  j = 0;
  for (i = 0; i < MAX_READERS; i++) {
    pthread_create(&tids[j], NULL, reader, &ids[j]);
    j++;
  }

  for (i = 0; i < MAX_WRITERS; i++) {
    pthread_create(&tids[j], NULL, writer, &ids[j]);
    j++;
  }

  for (i = 0; i < (MAX_READERS + MAX_WRITERS); i++) {
     pthread_join(tids[i], NULL);
  }

}
