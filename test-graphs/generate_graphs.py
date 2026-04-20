#!/usr/bin/env python3
"""
Generate medium and large test graphs for parallel BFS benchmarking.

Graph types:
  - Sparse random (Erdos-Renyi style, low avg degree)
  - Dense random (Erdos-Renyi style, high avg degree)
  - Chain/path (worst-case BFS depth)
  - Power-law / RMAT-like (skewed degree distribution, realistic)

Output format: .edgelist
    <num_vertices> <num_edges>
    <src> <dst>
    ...

Edges are undirected (listed once). Vertices are 0-indexed.
"""

import random
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def write_graph(filename, num_vertices, edges, description=""):
    """Write an edge list to a file."""
    filepath = os.path.join(SCRIPT_DIR, filename)
    with open(filepath, "w") as f:
        if description:
            for line in description.strip().split("\n"):
                f.write(f"# {line}\n")
        f.write(f"{num_vertices} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")
    print(f"  Written: {filename} ({num_vertices} vertices, {len(edges)} edges)")


def generate_sparse_random(num_vertices, avg_degree, seed=42):
    """Generate a sparse Erdos-Renyi-like random graph."""
    random.seed(seed)
    target_edges = (num_vertices * avg_degree) // 2
    edge_set = set()
    while len(edge_set) < target_edges:
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)
        if u != v:
            edge = (min(u, v), max(u, v))
            edge_set.add(edge)
    return sorted(edge_set)


def generate_dense_random(num_vertices, avg_degree, seed=123):
    """Generate a denser random graph."""
    return generate_sparse_random(num_vertices, avg_degree, seed)


def generate_chain(num_vertices):
    """Generate a path/chain graph (worst-case BFS depth = V-1)."""
    edges = [(i, i + 1) for i in range(num_vertices - 1)]
    return edges


def generate_powerlaw(num_vertices, num_edges, seed=999):
    """
    Generate a power-law degree distribution graph using preferential attachment
    (Barabasi-Albert model). This produces realistic social-network-like graphs
    with a few high-degree hubs and many low-degree nodes.
    """
    random.seed(seed)
    # Number of edges each new node adds
    m = max(1, (2 * num_edges) // num_vertices)
    m = min(m, 20)  # cap to keep it reasonable

    # Start with a small clique
    adj = [[] for _ in range(num_vertices)]
    edge_set = set()
    initial_size = m + 1

    for i in range(initial_size):
        for j in range(i + 1, initial_size):
            edge_set.add((i, j))
            adj[i].append(j)
            adj[j].append(i)

    # Degree list for preferential attachment (repeated entries = higher prob)
    degree_list = []
    for i in range(initial_size):
        degree_list.extend([i] * len(adj[i]))

    for new_node in range(initial_size, num_vertices):
        # Pick m distinct targets by preferential attachment
        targets = set()
        attempts = 0
        while len(targets) < m and attempts < m * 10:
            t = degree_list[random.randint(0, len(degree_list) - 1)]
            if t != new_node:
                targets.add(t)
            attempts += 1

        for t in targets:
            edge = (min(new_node, t), max(new_node, t))
            if edge not in edge_set:
                edge_set.add(edge)
                adj[new_node].append(t)
                adj[t].append(new_node)
                degree_list.append(new_node)
                degree_list.append(t)

    return sorted(edge_set)


def main():
    print("Generating medium graphs...")

    # Medium sparse: 1K vertices, avg degree ~6
    edges = generate_sparse_random(1000, 6)
    write_graph("medium_sparse.edgelist", 1000, edges,
                "Sparse random graph, 1K vertices, avg degree ~6")

    # Medium dense: 1K vertices, avg degree ~100
    edges = generate_dense_random(1000, 100)
    write_graph("medium_dense.edgelist", 1000, edges,
                "Dense random graph, 1K vertices, avg degree ~100")

    # Medium chain: 1K vertices, path graph
    edges = generate_chain(1000)
    write_graph("medium_chain.edgelist", 1000, edges,
                "Chain/path graph, 1K vertices, worst-case BFS depth")

    print("\nGenerating large graphs...")

    # Large sparse: 100K vertices, avg degree ~10
    edges = generate_sparse_random(100_000, 10)
    write_graph("large_sparse.edgelist", 100_000, edges,
                "Sparse random graph, 100K vertices, avg degree ~10")

    # Large dense: 100K vertices, avg degree ~100
    edges = generate_dense_random(100_000, 100)
    write_graph("large_dense.edgelist", 100_000, edges,
                "Dense random graph, 100K vertices, avg degree ~100")

    # Large power-law: 100K vertices, ~1M edges (Barabasi-Albert)
    edges = generate_powerlaw(100_000, 1_000_000)
    write_graph("large_powerlaw.edgelist", 100_000, edges,
                "Power-law (Barabasi-Albert) graph, 100K vertices\n"
                "Skewed degree distribution (few hubs, many low-degree nodes)\n"
                "Realistic for social network / web graph benchmarking")

    print("\nDone! All graphs written to:", SCRIPT_DIR)


if __name__ == "__main__":
    main()
