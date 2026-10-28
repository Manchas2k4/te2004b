from collections import deque
import heapq

class Stack:
    def __init__(self):
        self._data = []

    def empty(self):
        return not self._data

    def clear(self):
        self._data.clear()

    def push(self, element):
        self._data.append(element)

    def pop(self):
        if self.empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def top(self):
        if self.empty():
            raise IndexError("top from empty stack")
        return self._data[-1]

    def __str__(self):
        return f"Stack({self._data})"

class Queue:
    def __init__(self):
        self._data = deque()

    def empty(self):
        return not self._data

    def clear(self):
        self._data.clear()

    def enqueue(self, element):
        """Adds an element to the back of the queue."""
        self._data.append(element)

    def dequeue(self):
        """Removes and returns the element at the front of the queue."""
        if self.empty():
            raise IndexError("dequeue from empty queue")
        return self._data.popleft()

    def front(self):
        """Returns the element at the front of the queue without removing it."""
        if self.empty():
            raise IndexError("front from empty queue")
        return self._data[0]

    def __str__(self):
        return f"Queue({list(self._data)})"

class PriorityQueue:
    def __init__(self):
        self._data = []
        self._index = 0  # Desempate para prioridades idénticas (FIFO)

    def empty(self):
        return not self._data

    def clear(self):
        self._data.clear()
        self._index = 0

    def push(self, priority, value):
        """Inserts an element with a given priority. Lower priority values are popped first."""
        heapq.heappush(self._data, (priority, self._index, value))
        self._index += 1

    def pop(self):
        """Removes and returns the element with the highest priority (lowest priority value)."""
        if self.empty():
            raise IndexError("pop from empty priority queue")
        
        # Desestructuramos para retornar solo el (priority, value) o únicamente el value
        priority, _, value = heapq.heappop(self._data)
        return priority, value

    def top(self):
        """Returns the element with the highest priority without removing it."""
        if self.empty():
            raise IndexError("top from empty priority queue")
        
        priority, _, value = self._data[0]
        return priority, value

    def __str__(self):
        return f"PriorityQueue({[(p, v) for p, _, v in self._data]})"