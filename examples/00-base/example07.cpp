// =================================================================
//
// File: example07.cpp
// Author: Pedro Perez
// Description: This file implements the merge sort algorithm. The
//				time this implementation takes will be used as the
//				basis to calculate the improvement obtained with
//				parallel technologies.
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include "utils.h"

using namespace std;
using namespace std::chrono;

#define SIZE 10000000 //1e7

// =================================================================
// Swaps the contents of two locations in an array.
//
// @param a, an array of integer numbers.
// @param i, the index of an array position.
// @param j, the index of an array position.
// =================================================================
void swap(int *a, int i, int j) {
    int aux = a[i];
    a[i] = a[j];
    a[j] = aux;
}

// =================================================================
// Copy the contents of the range between the low and high indices 
// of array B to array A
//
// @param A, an array of integer numbers.
// @param B, an array of integer numbers.
// @param low, the lower limit.
// @param high, the upper limit.
// =================================================================
void copy_array(int *A, int *B, int low, int high) {
    int size = high - low + 1;
    memcpy(A + low, B + low, sizeof(int) * size);
}

// =================================================================
// Take two sorted segments of the array A as input and produce a 
// a segment of the array B as output, containing all the elements of 
// the inputs segments of A in sorted order. 
//
// @param A, an array of integer numbers.
// @param B, a temporal array of integer numbers.
// @param low, the lower limit.
// @param mid, the middle limit.
// @param high, the upper limit.
// =================================================================
void merge(int *A, int *B, int low, int mid, int high) {
    int i, j, k;
    i = low;
    j = mid + 1;
    k = low;
    while(i <= mid && j <= high){
        if(A[i] < A[j]){
            B[k] = A[i];
            i++;
        }else{
            B[k] = A[j];
            j++;
        }
        k++;
    }
    for(; j <= high; j++){
        B[k++] = A[j];
    }

    for(; i <= mid; i++){
        B[k++] = A[i];
    }
}


// =================================================================
// Recursively divide the array into two segments of (roughly) equal 
// length, until each segment contains only one element. Repeatedly 
// merge segments to create a new segment until the single array 
// contains all the sorted elements.
//
// @param A, an array of integer numbers.
// @param B, a temporal array of integer numbers.
// @param low, the lower limit.
// @param high, the upper limit.
// =================================================================
void split(int *A, int *B, int low, int high) {
    int  mid, size, i, j;

    if ((high - low + 1) == 1) {
        return;
    }

    mid = low + ((high - low) / 2);
    split(A, B, low, mid);
    split(A, B, mid +1, high);
    merge(A, B,low, mid, high);
    copy_array(A, B, low, high);
}

// =================================================================
// Sorts the elements, in ascending order, using the merge-sort 
// algorithm.
//
// @param A, an array of integer numbers to be sorted.
// @param size, the size of the array.
// =================================================================
void merge_sort(int *A, int size) {
    int *B = new int[size];
    split(A, B, 0, size - 1);
    delete [] B;
}

int main(int argc, char* argv[]) {
    // We will use pointers to handle large arrays.
    int *array, *aux;

    // These variables are used to keep track of the execution time.
    high_resolution_clock::time_point start, end;
    double timeElapsed;

    // We create the array and initialize it with random numbers.
    array = new int[SIZE];
    random_array(array, SIZE);
    display_array("before", array);

    aux = new int[SIZE];

    // We execute the task at least 10 times (N). It is necessary 
    // to do so, since it allows us to reduce the impact of the 
    // load on the operating system at the time of execution.
    cout << "Starting...\n";
    timeElapsed = 0;
    for (int j = 0; j < N; j++) {
        memcpy(aux, array, sizeof(int) * SIZE);

        // We take a clock record before execution.
        start = high_resolution_clock::now();

        // We perform the task.
        merge_sort(aux, SIZE);

        // We take a clock record after execution. We calculate the 
        // difference between the two records. This difference is 
        // the time it took to execute the task.
        end = high_resolution_clock::now();
        timeElapsed += 
            duration<double, std::milli>(end - start).count();
    }
    // We display the result and the average execution time.
    memcpy(array, aux, sizeof(int) * SIZE);
    display_array("after", array);
    cout << "avg time = " << fixed << setprecision(3) 
         << (timeElapsed / N) <<  " ms\n";

    // We delete all the allocated memory.
    delete [] array;
    delete [] aux;
    
    return 0;
}
