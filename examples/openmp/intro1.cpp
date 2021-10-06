// =================================================================
//
// File: intro1.cpp
// Author: Pedro Perez
// Description: This file shows the use of basic OpenMP directives.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[]) {
	#pragma omp parallel
	{
		cout << "Hello world!\n";
		cout << "¡Hola mundo!\n";
	}
	return 0;
}
