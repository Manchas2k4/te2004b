// =================================================================
//
// File: example01.cpp
// Author: Pedro Perez
// Description: This file contains the code that adds all the
//				elements of an integer array using pthreads.
//              To compile: g++ example01.cpp -lpthread
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <pthread.h>
#include "utils.h"

using namespace std;

const int SIZE = 100000000; //1e8
const int THREADS = 4;

typedef struct {
  int start, end; // [start, end)
  int *arr;
} Block;

void* sumArray(void* param) {
  double *acum;
  Block *block;
  int i;

  block = (Block *) param;
  acum =  new double;
  (*acum) = 0;
  for (i = block->start; i < block->end; i++) {
  
    (*acum) += block->arr[i];
  }
  return ( (void**) acum );
}

int main(int argc, char* argv[]) {
	int *a, block_size, i, j;
	double ms, result, *acum;
  Block blocks[THREADS];
  pthread_t tids[THREADS];

	a = new int[SIZE];
	fill_array(a, SIZE);
	display_array("a", a);

  block_size = SIZE / THREADS;
  for (i = 0; i < THREADS; i++) {
      blocks[i].arr = a;
      blocks[i].start = i * block_size;
      if (i != (THREADS - 1)) {
          blocks[i].end = (i + 1) * block_size;
      } else {
          blocks[i].end = SIZE;
      }
  }

	std::cout << "Starting...\n";
	ms = 0;
    for (j = 0; j < N; j++) {
        start_timer();

        result = 0;
        for (i = 0; i < THREADS; i++) {
            pthread_create(&tids[i], NULL, sumArray, (void*) &blocks[i]);
        }
        for (i = 0; i < THREADS; i++) {
            pthread_join(tids[i], (void**) &acum);
            result += (*acum);
            delete acum;
        }

        ms += stop_timer();
    }
	std::cout << "sum = " << setprecision(5) << result << "\n";
	std::cout << "avg time =  " << setprecision(5) << (ms / N) << "\n";

	delete [] a;
	return 0;
}
