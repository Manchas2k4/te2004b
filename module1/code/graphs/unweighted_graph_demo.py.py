"""
This demonstration uses the graph shown in the image 
images\\UnweightedGraph.png.
"""
from utils.graph import Graph
from utils.graph_traversal import dfs, bfs


if __name__ == "__main__":
    g = Graph()

    g.add_edge(1, 2)
    g.add_edge(2, 4)
    g.add_edge(3, 2)
    g.add_edge(4, 1)
    g.add_edge(4, 3)
    g.add_edge(4, 5)
    g.add_edge(5, 2)
    g.add_edge(5, 4)
    g.add_edge(5, 6)
    g.add_edge(6, 1)

    print("Graph")
    print(g)

    print("\nDFS starting from a source vertex.")
    path, visited, prev, steps = dfs(g, 6)
    print(f"Path = {path}")
    print(f"visited = {visited}")
    print(f"Prev = {prev}")
    print(f"Steps = {steps}")

    print("\nDFS between a source vertex (6) and a destination vertex (3).")
    path, visited, prev, steps = dfs(g, 6, 3)
    print(f"Path = {path}")
    print(f"visited = {visited}")
    print(f"Prev = {prev}")
    print(f"Steps = {steps}")

    print("\nBFS starting from a source vertex.")
    path, visited, prev, steps = bfs(g, 6)
    print(f"Path = {path}")
    print(f"visited = {visited}")
    print(f"Prev = {prev}")
    print(f"Steps = {steps}")

    print("\nBFS between a source vertex (6) and a destination vertex (3).")
    path, visited, prev, steps = bfs(g, 6, 3)
    print(f"Path = {path}")
    print(f"visited = {visited}")
    print(f"Prev = {prev}")
    print(f"Steps = {steps}")
