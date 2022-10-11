## =================================================================
##
## File: intro01.cpp
## Author: Pedro Perez
## Description: This file contains an example of threads in Python. 
##
## Copyright (c) 2022 by Tecnologico de Monterrey.
## All Rights Reserved. May be reproduced for any non-commercial
## purpose.
##
## =================================================================

import multiprocessing as mp
import threading as th

# CORES = multiprocessingmp.cpu_count() 
CORES = 4
LIMIT = 3

def task():
    id = th.current_thread().ident
    for i in range(0, LIMIT):
        print("Process id = ", id, " i = " , i)

if __name__ == "__main__":
    t = th.Thread(target=task)
    t.start()
    print("Main    : before joining process 0.")
    t.join()
    print("Main    : process 0 done.")
