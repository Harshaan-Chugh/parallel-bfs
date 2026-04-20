#!/usr/bin/env python3
"""
Reference BFS implementation for validating GPU BFS results.

Usage:
    python3 validate_bfs.py <graph.edgelist> <source_vertex> [output_file]

If output_file is provided, reads BFS depths from it and compares against
the reference. Output file should have one depth per line (vertex 0 on line 1).
Unreachable vertices should have depth -1.

Without output_file, just prints the reference BFS depths.
"""

import sys
from collections import deque


def load_graph(filename):
    """Load an .edgelist file into an adjacency list."""
    with open(filename) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    n, m = map(int, lines[0].split())
    adj = [[] for _ in range(n)]

    for line in lines[1:]:
        u, v = map(int, line.split())
        adj[u].append(v)
        adj[v].append(u)

    return n, adj


def bfs(adj, source):
    """Run BFS and return depth array."""
    n = len(adj)
    depth = [-1] * n
    depth[source] = 0
    queue = deque([source])

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                queue.append(v)

    return depth


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <graph.edgelist> <source_vertex> [output_file]")
        sys.exit(1)

    graph_file = sys.argv[1]
    source = int(sys.argv[2])

    n, adj = load_graph(graph_file)
    ref_depth = bfs(adj, source)

    if len(sys.argv) >= 4:
        # Validation mode
        output_file = sys.argv[3]
        with open(output_file) as f:
            gpu_depth = [int(l.strip()) for l in f if l.strip()]

        if len(gpu_depth) != n:
            print(f"FAIL: expected {n} depths, got {len(gpu_depth)}")
            sys.exit(1)

        mismatches = 0
        for i in range(n):
            if gpu_depth[i] != ref_depth[i]:
                if mismatches < 10:
                    print(f"  MISMATCH vertex {i}: expected {ref_depth[i]}, got {gpu_depth[i]}")
                mismatches += 1

        if mismatches == 0:
            print(f"PASS: all {n} vertices match reference BFS from source {source}")
        else:
            print(f"FAIL: {mismatches}/{n} mismatches")
            sys.exit(1)
    else:
        # Print mode
        print(f"BFS from source {source} on {graph_file} ({n} vertices):")
        if n <= 50:
            for i, d in enumerate(ref_depth):
                print(f"  vertex {i}: depth {d}")
        else:
            reachable = sum(1 for d in ref_depth if d >= 0)
            max_depth = max(ref_depth)
            print(f"  Reachable: {reachable}/{n}")
            print(f"  Max depth: {max_depth}")
            print(f"  First 20 depths: {ref_depth[:20]}")


if __name__ == "__main__":
    main()
