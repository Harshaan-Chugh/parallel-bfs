# Parallel BFS: Linear Algebraic vs Graph-First

Comparing two GPU-accelerated BFS approaches:
- **Linear Algebraic** (`linear-algebraic/`): BFS via SpMV on CSR adjacency matrix
- **Graph-First** (`graph-first/`): Direction-optimizing BFS (Beamer et al.)

## Quick Start

```bash
# Build both
make

# Test on small graphs (correctness)
./run.sh test-tiny

# Validate against reference BFS
./run.sh validate

# Test on all sizes
./run.sh test-all

# Benchmark on large graphs
./run.sh bench
```

## Commands

| Command | Description |
|---------|-------------|
| `make` | Build both into `build/` |
| `make linalg` | Build only linear-algebraic |
| `make graphfirst` | Build only graph-first |
| `make clean` | Remove `build/` |
| `./run.sh build` | Same as `make` |
| `./run.sh test-tiny` | Run on tiny graphs |
| `./run.sh test-medium` | Run on medium (1K vertex) graphs |
| `./run.sh test-large` | Run on large (100K vertex) graphs |
| `./run.sh test-all` | Run on everything |
| `./run.sh validate` | Run + check correctness vs Python BFS |
| `./run.sh bench` | Benchmark both on large graphs |

## Running a Single Graph

```bash
./build/bfs_linalg test-graphs/tiny_cycle.edgelist 0
./build/bfs_graphfirst test-graphs/large_powerlaw.edgelist 0
```

## Generating Test Graphs

```bash
cd test-graphs && python3 generate_graphs.py
```

## Project Structure

```
parallel-bfs/
├── Makefile
├── run.sh
├── build/                  # compiled binaries (gitignored)
├── linear-algebraic/
│   └── bfs_linalg.cu       # SpMV-based BFS
├── graph-first/
│   └── bfs_graphfirst.cu   # direction-optimizing BFS
└── test-graphs/
    ├── graph_loader.h       # shared C graph loader (CSR)
    ├── generate_graphs.py   # generates medium/large graphs
    ├── validate_bfs.py      # reference BFS for validation
    ├── tiny_*.edgelist      # small hand-written graphs
    ├── medium_*.edgelist    # 1K vertex graphs
    └── large_*.edgelist     # 100K vertex graphs
```

## GPU Note

All scripts use `CUDA_VISIBLE_DEVICES=0` to restrict to 1 GPU.
