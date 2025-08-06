// =================================================================
//
// File: utils.h
// Author: Pedro Perez
// Description: This file contains the implementation of the
//				functions used to take the time and perform the
//				speed up calculation; as well as functions for
//				initializing integer arrays.
//
// Copyright (c) 2024 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#ifndef UTILS_H
#define UTILS_H

#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>

#define N 			10
#define DISPLAY		50
#define TOP_VALUE	10000

// =================================================================
// Initializes an array with random values between 1 and TOP_VALUE.
//
// @param array, an array of integer numbers.
// @param size, the amount of numbers.
// =================================================================
void random_array(int *array, int size) {
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(1, TOP_VALUE);

	for (int i = 0; i < size; i++) {
		array[i] = distribution(generator);
	}
}

// =================================================================
// Initializes an array with consecutive values of 1 and TOP_VALUE
// across all locations.
//
// @param array, an array of integer numbers.
// @param size, the amount of numbers.
// =================================================================
void fill_array(int *array, int size) {
	for (int i = 0; i < size; i++) {
		array[i] = (i % TOP_VALUE) + 1;
	}
}

// =================================================================
// Displays the first N locations in the array.
//
// @param array, an array of integer numbers.
// @param size, the amount of numbers.
// =================================================================
void display_array(const char *text, int *array) {
	printf("%s = [%4i", text, array[0]);
	for (int i = 1; i < DISPLAY; i++) {
		printf(",%5i", array[i]);
	}
	printf(", ... ,]\n");
}
#endif