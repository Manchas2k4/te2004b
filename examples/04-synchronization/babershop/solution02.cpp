#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;

int chairs = 4;
int customers = 0;
mutex mutex_lock;
condition_variable customer_cond;
condition_variable barber_cond;

void customer(int id) {
    cout << "Customer " << id << " starting...\n";

    for (int i = 0; i < 5; i++) {
        unique_lock<mutex> lock(mutex_lock);

        cout << "Customer " << id << ": I'm going to the barbershop.\n";
        if (customers == chairs) {
            cout << "Customer " << id << ": There is no room, I'm leaving.\n";
            
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            continue;
        }
        customers++;
        customer_cond.notify_one();
        
        barber_cond.wait(lock);

        cout << "Customer " << id << ": The barber is attending to me.\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        cout << "Customer " << id << ": Ready! I'll be leaving.\n";
        
        customers--;
        lock.unlock();

        cout << "Customer " << id << ": I'll be back in a bit.\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
    cout << "Customer " << id << " ending...\n";
}

void barber() {
    cout << "Barber starting...\n";
    
    unique_lock<mutex> lock(mutex_lock);
    while (1) {
        while (customers == 0) {
            cout << "Barber: Waiting for a customer..\n";
            customer_cond.wait(lock);
        }

        cout << "Barber: I am attending to a client.\n";
        
        barber_cond.notify_one();
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        lock.lock();
    }
}

int main(int argc, char* argv[]) {
    thread barber_thread;
    thread customer_thread[7];

    barber_thread = thread(barber);

    for (int i = 0; i < 7; i++) {
        customer_thread[i] = thread(customer, (i + 1));
    }

    for (int i = 0; i < 7; i++) {
        customer_thread[i].join();
    }

    barber_thread.detach();

    return 0;
}