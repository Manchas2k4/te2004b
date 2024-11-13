#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>

using namespace std;

int chairs = 4;
int customers = 0;
bool shop_open = true;
mutex mutex_lock, customer_lock, barber_lock;
mutex customer_done_lock, barber_done_lock;

void customer(int id) {
	cout << "Customer " << id << " starting...\n";

	for (int i = 0; i < 5; i++) {
		mutex_lock.lock();	

		cout << "Customer " << id << ": I'm going to the barbershop.\n";
		
		if (customers == chairs) {
			cout << "Customer " << id << ": There is no room, I'm leaving.\n";
			mutex_lock.unlock();
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			continue;
		}
		customers++;
		mutex_lock.unlock();

		customer_lock.unlock();
		barber_lock.lock();

		cout << "Customer " << id << ": The barber is attending to me.\n";

		std::this_thread::sleep_for(std::chrono::milliseconds(500));

		customer_done_lock.unlock();
		barber_done_lock.lock();

		cout << "Customer " << id << ": Ready! I'll be leaving.\n";

		mutex_lock.lock();
		customers--;
		mutex_lock.unlock();

		cout << "Customer " << id << ": I'll be back in a bit.\n";

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
	
	cout << "Customer " << id << " ending...\n";
}

void barber() {
	cout << "Barber starting...\n";

	while(1) {
		cout << "Barber: Waiting for a customer..\n";

		customer_lock.lock();
		barber_lock.unlock();

		cout << "Barber: I am attending to a client.\n";
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));

		customer_done_lock.lock();
		barber_done_lock.unlock();
	}
}

int main(int argc, char* argv[]) {
	thread barber_thread;
	thread customer_thread[7];

	customer_lock.lock();
	barber_lock.lock();
	customer_done_lock.lock();
	barber_done_lock.lock();

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