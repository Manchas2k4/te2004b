## =================================================================
##
## File: intro01.cpp
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

# CORES = multiprocessingmp.cpu_count() 
CORES = 4
LIMIT = 3

def task():
    process = mp.current_process()
    for i in range(0, LIMIT):
        print("Process id = ", process.pid, " i = " , i)

if __name__ == "__main__":
    one_process = mp.Process(target=task)
    one_process.start()
    print("Main    : before joining process 0.")
    one_process.join()
    print("Main    : process 0 done.")
