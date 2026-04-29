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

## Perlmutter Results

The poster/report numbers are from `results/benchmark_20260428_042111.csv`
and summarized in `results/summary_20260428_042111.csv`. Each entry below is
the median of 10 single-GPU A100 runs, measured with CUDA events and excluding
graph loading / printing.

| Graph | Best SpMV variant | Best SpMV ms | Graph-first ms | Graph-first / SpMV |
|---|---:|---:|---:|---:|
| `medium_chain` | bitmap | 18.733 | 32.918 | 1.76x |
| `medium_dense` | warpbitmap | 0.557 | 0.697 | 1.25x |
| `medium_sparse` | warp | 0.665 | 0.758 | 1.14x |
| `large_dense` | bitmap | 4.438 | 5.085 | 1.15x |
| `large_powerlaw` | warpbitmap | 2.325 | 6.264 | 2.69x |
| `large_sparse` | bitmap | 1.446 | 1.927 | 1.33x |

Current takeaway: the best linear-algebraic SpMV kernel wins on these six
synthetic inputs, while graph-first is closest on random sparse/dense graphs
and loses the most on the power-law input because queue construction and
atomic frontier updates add overhead around hub-heavy frontiers.
