from utils.linear_containers import Stack, Queue
from utils.graph import Graph

def dfs(g, src, dest=None):
    if not g.has_vertex(src):
        raise ValueError(f"Source vertex '{src}' was not found.")

    if dest is not None and not g.has_vertex(dest):
        raise ValueError(f"Destination vertex '{dest}' was not found.")

    visited = []
    prev = {src : None}
    pending = Stack()
    path = []
    steps = 0

    pending.push(src)
    while not pending.empty():
        current = pending.top() 
        pending.pop()
        
        if current in visited:
            continue

        visited.append(current)
        steps += 1

        if dest is not None and current == dest:
            break

        for neighbor in g.get_neighbors(current):
            if neighbor not in visited:
                if neighbor not in prev:
                    prev[neighbor] = current
                pending.push(neighbor)

    if dest is not None and dest in visited:
        current = dest
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()

    return path, visited, prev, steps


def bfs(g, src, dest=None):
    if not g.has_vertex(src):
        raise ValueError(f"Source vertex '{src}' was not found.")

    if dest is not None and not g.has_vertex(dest):
        raise ValueError(f"Destination vertex '{dest}' was not found.")

    visited = []
    prev = {src : None}
    pending = Queue()
    path = []
    steps = 0

    pending.enqueue(src)
    while not pending.empty():
        current = pending.front() 
        pending.dequeue()
        
        if current in visited:
            continue

        visited.append(current)
        steps += 1

        if dest is not None and current == dest:
            break

        for neighbor in g.get_neighbors(current):
            if neighbor not in visited:
                if neighbor not in prev:
                    prev[neighbor] = current
                pending.enqueue(neighbor)

    if dest is not None and dest in visited:
        current = dest
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()

    return path, visited, prev, steps