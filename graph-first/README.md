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

## Runtime Results

These runtimes are medians of 10 runs on a single NVIDIA A100 GPU on NERSC
Perlmutter. The raw data is in `../results/benchmark_20260428_042111.csv`, with
medians in `../results/summary_20260428_042111.csv`.

| Graph | Avg Degree | Depth | Graph-first ms | Top-down launches | Bottom-up launches |
|-------|-----------:|------:|---------------:|------------------:|-------------------:|
| medium_chain | 2 | 999 | 32.92 | 1000 | 0 |
| medium_dense | 100 | 2 | 0.70 | 1 | 2 |
| medium_sparse | 6 | 6 | 0.76 | 4 | 3 |
| large_dense | 100 | 4 | 5.09 | 3 | 2 |
| large_powerlaw | 40 | 3 | 6.26 | 2 | 2 |
| large_sparse | 10 | 7 | 1.93 | 5 | 3 |

The chain graph is a worst case for direction optimization because each level
has almost no frontier parallelism. On dense and sparse random graphs, the
hybrid stays close to the best SpMV variant; on the power-law graph, queue
construction and atomic updates around hub-heavy frontiers dominate more of the
runtime.
