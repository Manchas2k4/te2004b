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
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;

#define MAX_READERS     5
#define MAX_WRITERS     2
#define MAX_TIMES       5

mutex mtx;
condition_variable cond_var;

int readers;
bool writer_active;

/************* CODE FOR READERS *************/
void beginReading(int id) {
  unique_lock<mutex> lock(mtx);
  
  cout << "Reader " << id << " is waiting to read\n";
  cond_var.wait(lock, [] () { return writer_active == false; });

  readers++;
  cout << "Reader " << id << " is now reading\n";
}

void endReading(int id) {
  unique_lock<mutex> lock(mtx);

  readers--;
  if (readers == 0) {
    cond_var.notify_all();
  }
}

void reader(int id) {
  for (int i = 0; i < MAX_TIMES; i++) {
    beginReading(id);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    endReading(id);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}
/************* CODE FOR READERS *************/


/************* CODE FOR WRITERS *************/
void beginWriting(int id) {
  unique_lock<std::mutex> lock(mtx);
  cond_var.wait(lock, [] () { return (writer_active == false) && (readers == 0); });

  writer_active = true;
  cout << "Writer " << id << " is writing\n";
}

void endWriting(int id) {
  unique_lock<std::mutex> lock(mtx);
  writer_active = false;
  cond_var.notify_all();
}

void writer(int id) {
  for (int i = 0; i < MAX_TIMES; i++) {
    beginWriting(id);
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    endWriting(id);
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  }
}
/************* CODE FOR WRITERS *************/

int main(int argc, char* argv[]) {
  int j;
  thread threads[MAX_READERS + MAX_WRITERS];

  readers = 0;
  writer_active = false;

  j = 0;
  for (int i = 0; i < MAX_READERS; i++) {
    threads[j] = thread(reader, i);
    j++;
  }

  for (int i = 0; i < MAX_WRITERS; i++) {
    threads[j] = thread(writer, i);
    j++;
  }

  for (int i = 0; i < (MAX_READERS + MAX_WRITERS); i++) {
    threads[i].join();
  }

}
