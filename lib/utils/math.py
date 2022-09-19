import random
from typing import List


def min_degree(graph):
    return min(len(v) for v in graph.values())


def vertices_with_min_degree(graph):
    degrees = {k: len(v) for k, v in graph.items()}
    min_degree = min(degrees.values())
    return [k for k, v in degrees.items() if v == min_degree]


def vertices_with_min_rel_degree(graph, seq):
    degrees = {key: sum(v in seq for v in value) for key, value in graph.items()}
    min_degree = min(degrees.values())
    return [k for k, v in degrees.items() if v == min_degree]


def get_sequences_which_covers_all_elements_pairs(n: int, m: int) -> List[List]:
    """
    Return minimum amount of combinations with length m sampled from n elements.
    It's guaranteed that any pair of elements be at least in one combination.

    Args:
        n: num elements
        m: set length
    Return:
        List of combinations
    """
    out = []
    graph = {i: set() for i in range(n)}
    while min_degree(graph) != n - 1:
        m_seq = []
        for i in range(m):
            filtered_graph = {k: v for k, v in graph.items() if k not in m_seq}
            verts_rel = vertices_with_min_rel_degree(filtered_graph, m_seq)
            selected_graph = {k: v for k, v in filtered_graph.items() if k in verts_rel}
            verts = vertices_with_min_degree(selected_graph)
            new_vertex = random.choice(verts)

            m_seq.append(new_vertex)
            for v in m_seq[:-1]:
                graph[v] |= {new_vertex}
                graph[new_vertex] |= {v}
        out.append(sorted(m_seq))
    return out
