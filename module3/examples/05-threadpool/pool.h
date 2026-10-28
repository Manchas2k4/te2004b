#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <iostream>
#include <iomanip>
#include <queue> 
#include <thread>
#include <mutex> 
#include <condition_variable> 
#include <functional> 

typedef unsigned int uint;

using namespace std;

class ThreadPool {
private:
    vector<thread> threads;
    bool stop;
    queue<function<void()> > tasks;
    mutex queue_mutex;
    condition_variable cv; 

public:
    ThreadPool(uint);
    ~ThreadPool();
    void enqueue(function<void()>);
};

ThreadPool::ThreadPool(uint num_threads = thread::hardware_concurrency())
    : threads(vector<thread>(num_threads)), stop(false) {
    for (uint i = 0; i < num_threads; i++) {
        threads[i] = thread([this] {
            while(true) {
                function<void()> task;
                // The reason for putting the below code here is 
                // to unlock the queue before executing the task 
                // so that other threads can perform enqueue tasks 
                {
                    // Locking the queue so that data can be 
                    // shared safely
                    unique_lock<mutex> lock(queue_mutex); 
                    // Waiting until there is a task to execute 
                    // or the pool is stopped 
                    cv.wait(lock, [this] { 
                        return !tasks.empty() || stop; 
                    });

                    // Exit the thread in case the pool is 
                    // stopped and there are no tasks
                    if (stop && tasks.empty()) { 
                        return; 
                    }

                    // Get the next task from the queue 
                    task = move(tasks.front()); 
                    tasks.pop();  
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    { 
        // Lock the queue to update the stop flag safely 
        unique_lock<mutex> lock(queue_mutex); 
        stop = true; 
    }

    // Notify all threads 
    cv.notify_all(); 

    // Joining all worker threads to ensure they have completed 
    // their tasks 
    for (auto& thread : threads) { 
        thread.join(); 
    } 
}

void ThreadPool::enqueue(function<void()> task) { 
    { 
        unique_lock<std::mutex> lock(queue_mutex); 
        tasks.emplace(move(task)); 
    } 
    cv.notify_one(); 
} 

#endif