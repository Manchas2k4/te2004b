#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <pthread.h>

const int THREADS = 4;
const int LIMIT = 5;

using namespace std;

void* task(void* param) {
    int i, id;

    id = *((int*) param);
    for (i = 0; i < LIMIT; i++) {
        cout << "PID = " << getpid() << " ID = " << id
            << " i = " << i << "\n";
    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    pthread_t tid[THREADS];
    int i;

    for (i = 0; i < THREADS; i++) {
        pthread_create(&tid[i], NULL, task, (void*) &i);
    }

    for (i = 0; i < THREADS; i++) {
        pthread_join(tid[i], NULL);
    }

    return 0;
}
