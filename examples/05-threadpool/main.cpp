#include <iostream>
#include <iomanip>
#include <chrono>
#include "pool.h"

using namespace std;

int main(int argc, char* argv[]) { 
    // Create a thread pool with 4 threads 
    ThreadPool pool(4); 
  
    // Enqueue tasks for execution 
    for (int i = 0; i < 20; ++i) { 
        pool.enqueue([i] { 
            cout << "Task " << i << " is running on thread "
                 << this_thread::get_id() << endl; 
            // Simulate some work 
            this_thread::sleep_for( 
                chrono::milliseconds(100)); 
        }); 
    } 
  
    return 0; 
}