"""
This demonstration uses the graph shown in the image 
images\\Robot-navigation.png.
"""
from utils.graph import Graph
from utils.graph_traversal import dfs, bfs

if __name__ == "__main__":
    g = Graph(directed=False)

    # ROW 1
    g.add_edge((1, 1), (1, 2))
    g.add_edge((1, 1), (2, 1))

    g.add_edge((1, 2), (1, 3))

    g.add_edge((1, 3), (2, 3))

    g.add_edge((1, 5), (1, 6))
    g.add_edge((1, 5), (2, 5))

    g.add_edge((1, 6), (1, 7))

    g.add_edge((1, 7), (1, 8))

    g.add_edge((1, 8), (2, 8))

    # ROW 2
    g.add_edge((2, 1), (3, 1))

    g.add_edge((2, 3), (3, 3))

    g.add_edge((2, 5), (3, 5))

    g.add_edge((2, 8), (3, 8))

    # ROW 3
    g.add_edge((3, 1), (4, 1))

    g.add_edge((3, 3), (3, 4))

    g.add_edge((3, 4), (3, 5))

    g.add_edge((3, 5), (4, 5))

    g.add_edge((3, 8), (4, 8))

    # ROW 4
    g.add_edge((4, 1), (5, 1))

    g.add_edge((4, 5), (4, 6))
    g.add_edge((4, 5), (5, 5))

    g.add_edge((4, 6), (4, 7))

    g.add_edge((4, 7), (4, 8))

    g.add_edge((4, 8), (5, 8))

    # ROW 5
    g.add_edge((5, 1), (5, 2))

    g.add_edge((5, 2), (5, 3))

    g.add_edge((5, 3), (5, 4))

    g.add_edge((5, 4), (5, 5))
    g.add_edge((5, 4), (6, 4))

    g.add_edge((5, 8), (6, 8))

    # ROW 6
    g.add_edge((6, 4), (7, 4))

    g.add_edge((6, 6), (6, 7))
    g.add_edge((6, 6), (7, 6))

    g.add_edge((6, 7), (6, 8))

    g.add_edge((6, 8), (7, 8))

    # ROW 7
    g.add_edge((7, 1), (7, 2))
    g.add_edge((7, 1), (8, 1))

    g.add_edge((7, 2), (7, 3))

    g.add_edge((7, 3), (7, 4))

    g.add_edge((7, 4), (7, 5))

    g.add_edge((7, 5), (7, 6))

    g.add_edge((7, 6), (8, 6))

    g.add_edge((7, 8), (8, 8))

    # ROW 8
    g.add_edge((8, 6), (8, 7))

    g.add_edge((8, 7), (8, 8))

    start = (1, 1)
    end = (8, 8)

    print("\nDFS between a source vertex ((1,1)) and a destination vertex ((8,8)).")
    path, visited, prev, steps = dfs(g, start, end)
    print(f"Path = {path}")
    print(f"visited = {visited}")
    print(f"Prev = {prev}")
    print(f"Steps = {steps}")

    print("\nBFS between a source vertex ((1,1)) and a destination vertex ((8,8)).")
    path, visited, prev, steps = bfs(g, start, end)
    print(f"Path = {path}")
    print(f"visited = {visited}")
    print(f"Prev = {prev}")
    print(f"Steps = {steps}")