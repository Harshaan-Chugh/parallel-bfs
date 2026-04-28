# Parallel BFS: Linear Algebraic vs Graph-First

Comparing two GPU-accelerated BFS approaches on the same CSR graph loader:
- **Linear Algebraic** (`linear-algebraic/`): BFS via masked SpMV on a CSR adjacency matrix
- **Graph-First** (`graph-first/`): Direction-optimizing BFS with queue push and bitmap pull kernels

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

# Produce a CSV for the poster/report
python3 scripts/benchmark.py --repeats 10 --groups medium large
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

# Linear-algebraic variants
./build/bfs_linalg test-graphs/large_sparse.edgelist 0 baseline
./build/bfs_linalg test-graphs/large_sparse.edgelist 0 warp
./build/bfs_linalg test-graphs/large_sparse.edgelist 0 bitmap
./build/bfs_linalg test-graphs/large_sparse.edgelist 0 pushpull
./build/bfs_linalg test-graphs/large_sparse.edgelist 0 warpbitmap

# Graph-first alpha/beta switch tuning
./build/bfs_graphfirst test-graphs/large_sparse.edgelist 0 --alpha 14 --beta 24
```

Both binaries support `--dump` to print one raw depth per vertex for validation.

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
├── scripts/
│   ├── benchmark.py        # repeat benchmarks and write CSV
│   └── perlmutter_bench.slurm
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

On Perlmutter, submit the batch benchmark with:

```bash
sbatch -A <your_nersc_account> scripts/perlmutter_bench.slurm
```

The batch job builds both binaries, restricts execution to one GPU, and writes
CSV timing data under `results/`.
