# Graph-First Direction-Optimizing BFS

This implementation follows Beamer et al.'s direction-optimizing BFS idea using
explicit graph traversal kernels rather than a linear-algebraic SpMV framing.

## Kernels

- **Top-down push:** one thread processes one frontier vertex, scans its
  neighbors, and appends newly discovered vertices to a device queue with
  `atomicAdd`. `atomicCAS` on `visited[]` prevents duplicate discoveries.
- **Bottom-up pull:** one thread processes one unvisited vertex and checks its
  neighbors against a bitmap frontier. It stops as soon as one parent is found.
- **Direction switch:** the host keeps both a queue and bitmap for every
  frontier. It switches from top-down to bottom-up when the current frontier's
  edge volume exceeds `|E| / alpha`, then switches back when the frontier size
  drops below `|V| / beta`.

Default thresholds are `alpha = 14` and `beta = 24`, matching the common
Beamer heuristic.

## Usage

```bash
make graphfirst
./build/bfs_graphfirst test-graphs/tiny_cycle.edgelist 0
./build/bfs_graphfirst test-graphs/large_powerlaw.edgelist 0 --alpha 14 --beta 24
./build/bfs_graphfirst test-graphs/tiny_cycle.edgelist 0 --dump
```

The reported `BFS time` is measured with CUDA events around the traversal and
excludes graph loading and result printing.
