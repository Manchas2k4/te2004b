// =================================================================
//
// File: example02.cpp
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector using pthreads.
//              To compile: g++ example02.cpp -lpthread
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <climits>
#include <unistd.h>
#include <pthread.h>
#include "utils.h"

using namespace std;

const int RENS = 10000; //1e5
const int COLS = 10000;
const int THREADS = 4;

typedef struct {
  int start, end; // [start, end)
  int *m, *b, *c;
} Block;

void* matrixXVector(void* param) {
  Block *block;
  int i, j, acum;

  block = (Block *) param;
  for (i = block->start; i < block->end; i++) {
      acum = 0;
      for (j = 0; j < COLS; j++) {
          acum += (block->m[(i * COLS) + j] * block->b[i]);
      }
      block->c[i] = acum;
  }
  pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
	int i, j, *m, *b, *c, block_size;
	double ms;
  Block blocks[THREADS];
  pthread_t tids[THREADS];

	m = new int[RENS* COLS];
	b = new int[RENS];
	c = new int[RENS];

	for (i = 0; i < RENS; i++) {
		for (j = 0; j < COLS; j++) {
			m[(i * COLS) + j] = (j + 1);
		}
		b[i] = 1;
	}

  block_size = RENS / THREADS;
  for (i = 0; i < THREADS; i++) {
      blocks[i].start = i * block_size;
      if (i != (THREADS - 1)) {
          blocks[i].end = (i + 1) * block_size;
      } else {
          blocks[i].end = RENS;
      }
      blocks[i].m = m;
      blocks[i].b = b;
      blocks[i].c = c;
  }

	std::cout << "Starting...\n";
	ms = 0;
	for (j = 0; j < N; j++) {
        start_timer();

        for (i = 0; i < THREADS; i++) {
            pthread_create(&tids[i], NULL, matrixXVector, (void*) &blocks[i]);
        }
        for (i = 0; i < THREADS; i++) {
            pthread_join(tids[i], NULL);
        }

        ms += stop_timer();
	}
	display_array("c:", c);
	std::cout << "avg time =  " << setprecision(5) << (ms / N) << "\n";

	delete [] m;
	delete [] b;
	delete [] c;
	return 0;
}
