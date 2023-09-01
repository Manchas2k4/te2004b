// =================================================================
//
// File: intro01.cpp
// Author: Pedro Perez
// Description: This file shows some of the basic OpenMP directives.
//
// Copyright (c) 2022 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <omp.h>

int main(int argc, char* argv[]) {
	#pragma omp parallel
	{
		printf("Hello world!\n");
		printf("¡Hola mundo!\n");
	}
	return 0;
}
