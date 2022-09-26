## =================================================================
##
## File: intro04.cpp
## Author: Pedro Perez
## Description: This file contains an example of processes in Python. 
##
## Copyright (c) 2022 by Tecnologico de Monterrey.
## All Rights Reserved. May be reproduced for any non-commercial
## purpose.
##
## =================================================================

import multiprocessing as mp
import threading

# CORES = mp.cpu_count() 
CORES = 4

def task(id, limit):
    for i in range(0, limit):
        print("Process id = ", id, " i = " , i)

if __name__ == "__main__":
    processes = list()
    for index in range(CORES):
        process = mp.Process(target=task, args=(index, (index * 2), ))
        processes.append(process)
        process.start()

    for index, process in enumerate(processes):
        print("Main    : before joining process ", index, ".")
        process.join()
        print("Main    : process ", index, " done.")