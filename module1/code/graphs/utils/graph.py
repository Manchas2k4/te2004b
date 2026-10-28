import math

_UNSET = object()

class Graph:
    """Represents an weighted directed or undirected graph."""
    def __init__(self, directed=True, weighted=False):
        self._adjacency_list = {}
        self._directed = directed
        self._weighted = weighted

    @staticmethod
    def _normalize_weight(weight):
        if not isinstance(weight, (int, float)):
            raise TypeError(f"Edge weight must be a number (int or float), got {type(weight).__name__}.")

        if isinstance(weight, bool):
            raise TypeError(f"Invalid weight type '{type(weight).__name__}': boolean values are not accepted as edge weights.")
        
        if not math.isfinite(weight): 
            raise ValueError(f"Invalid weight '{weight}': weight must be a finite real number (cannot be NaN or Infinity).")

        if weight <= 0:
            raise ValueError(f"Invalid weight '{weight}': weight must be a positive number greater than zero.")

        return float(weight)

    def _require_weighted(self):
        """Ensures that the graph is weighted."""
        if not self._weighted:
            raise ValueError("This operation requires a weighted graph.")

    def has_vertex(self, v):
        """Indicates whether the vertex exists."""
        return v in self._adjacency_list

    def has_edge(self, u, v):
        """Indicates whether an edge exists from u to v."""
        return (u in self._adjacency_list
                and v in self._adjacency_list[u])

    def get_vertices(self):
        """Returns a list of the graph's vertices."""
        return list(self._adjacency_list.keys())

    def get_neighbors(self, v):
        """Returns a copy of the list of neighbors of v."""
        if v not in self._adjacency_list:
            raise ValueError(f"Vertex '{v}' was not found in the graph.")
        
        # Returns a dict copy {neighbor: weight} to prevent external mutation
        return self._adjacency_list[v].copy()

    def get_weight(self, u, v):
        """Returns the weight of an edge in a weighted graph."""
        self._require_weighted()

        if u not in self._adjacency_list:
            raise ValueError(f"Source vertex '{u}' was not found in the graph.")

        if v not in self._adjacency_list:
            raise ValueError(f"Destination vertex '{v}' was not found in the graph.")

        if v not in self._adjacency_list[u]:
            raise ValueError(f"Edge from '{u}' to '{v}' does not exist.")

        return self._adjacency_list[u][v]

    def add_vertex(self, v):
        """Adds a vertex if it does not exist."""
        if v in self._adjacency_list:
            raise ValueError(f"Vertex '{v}' already exists in the graph.")
        
        self._adjacency_list[v] = {}

    def remove_vertex(self, v):
        """Removes a vertex and all its edges."""
        if v not in self._adjacency_list:
            raise ValueError(f"Vertex '{v}' was not found in the graph.")

        # Remove incoming edges from all other vertices
        for neighbors in self._adjacency_list.values():
            neighbors.pop(v, None)

        del self._adjacency_list[v]

    def add_edge(self, u, v, weight=_UNSET):
        """Adds an edge; creates the vertices if necessary."""
        if u == v:
            raise ValueError(f"Cannot create a self-loop: source vertex '{u}' and destination vertex '{v}' are the same.")

        if self._weighted:
            if weight is _UNSET:
                raise ValueError("A weight must be provided for a weighted graph.")
            
            new_weight = self._normalize_weight(weight)
        else:
            if weight is not _UNSET:
                raise ValueError("A weight cannot be provided for an unweighted graph.")

            new_weight = None

        if self.has_edge(u, v):
            raise ValueError(f"Edge from '{u}' to '{v}' already exists.")
             
        if u not in self._adjacency_list:
            self._adjacency_list[u] = {}
            
        if v not in self._adjacency_list:
            self._adjacency_list[v] = {}

        self._adjacency_list[u][v] = new_weight
        if not self._directed:
            self._adjacency_list[v][u] = new_weight

    def update_weight(self, u, v, weight):
        """Updates the cost of the edge between u and v"""
        self._require_weighted()

        if not self.has_edge(u, v):
            raise ValueError(f"Edge from '{u}' to '{v}' does not exist.")

        new_weight = self._normalize_weight(weight)
        self._adjacency_list[u][v] = new_weight
        if not self._directed:
            self._adjacency_list[v][u] = new_weight

    def remove_edge(self, u, v):
        """Removes an edge if it exists."""
        if u not in self._adjacency_list:
            raise ValueError(f"Source vertex '{u}' was not found in the graph.")
            
        if v not in self._adjacency_list:
            raise ValueError(f"Destination vertex '{v}' was not found in the graph.")

        if v not in self._adjacency_list[u]:
            raise ValueError(f"Edge from '{u}' to '{v}' does not exist.")

        del self._adjacency_list[u][v]
        if not self._directed:
            del self._adjacency_list[v][u]

    def __str__(self):
        """Returns a string representation of the graph."""
        lines = []

        for vertex, neighbors in self._adjacency_list.items():
            if self._weighted:
                edges = ", ".join(
                    f"{neighbor}({weight})"
                    for neighbor, weight in neighbors.items()
                )
                lines.append(f"{vertex}\t-> [{edges}]")
            else:
                edges = ", ".join(map(str, neighbors.keys()))
                lines.append(f"{vertex}\t-> [{edges}]")

        return "\n".join(lines)    