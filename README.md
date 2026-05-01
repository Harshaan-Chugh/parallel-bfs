# parallel-bfs

**High-performance Breadth-First Search across GPU and distributed-memory architectures.**

This project implements and benchmarks BFS in two HPC settings:

1. **Single-node GPU BFS on CUDA/A100**
   - **Linear-Algebraic BFS**: expresses BFS as masked sparse matrix-vector multiplication over CSR graphs.
   - **Graph-First BFS**: implements direction-optimizing push/pull traversal with GPU frontiers.

2. **Multi-node CPU BFS with MPI on Perlmutter**
   - **1D partitioning**: distributes graph ownership by vertex ranges.
   - **2D partitioning**: explores communication-aware distributed graph partitioning.

The goal is to compare BFS design tradeoffs across architectures: **linear algebra vs graph traversal on GPUs**, and **partitioning/communication strategies in distributed-memory MPI**.

---

## Highlights

- Shared CSR graph loader for fair comparison between GPU implementations.
- Multiple SpMV-style BFS variants: `baseline`, `warp`, `bitmap`, `pushpull`, and `warpbitmap`.
- Graph-first CUDA BFS with tunable direction-switching parameters.
- MPI BFS binaries for 1D and 2D distributed traversal.
- Correctness validation against a Python reference BFS.
- Benchmark scripts for synthetic edgelist graphs and SNAP road networks.
- Raw result collection under `results/` for reproducibility.

---

## Repository Structure

```text
parallel-bfs/
├── linear-algebraic/        # CUDA SpMV-style BFS implementation
├── graph-first/             # CUDA direction-optimizing BFS implementation
├── mpi/                     # MPI BFS implementation sources
├── scripts/                 # Benchmark, validation, and Perlmutter batch scripts
├── test-graphs/             # Synthetic graph inputs + Python reference validator
├── snap-graphs/             # Downloaded SNAP road network inputs
├── results/                 # Raw benchmark logs, CSVs, and summaries
├── Makefile
└── run.sh
```

---

## Implementations

### Linear-Algebraic GPU BFS

The linear-algebraic implementation treats BFS as repeated sparse matrix-vector operations over a graph stored in CSR form. It explores several kernel variants to understand how frontier representation and memory access patterns affect performance.

Supported variants:

| Variant | Idea |
|---|---|
| `baseline` | straightforward masked traversal |
| `warp` | warp-oriented parallelism for neighbor checks |
| `bitmap` | compact frontier/visited representation |
| `pushpull` | switches between push and pull traversal modes |
| `warpbitmap` | combines warp-level traversal with bitmap state |

### Graph-First GPU BFS

The graph-first implementation uses a more traditional frontier-based BFS design. It supports direction optimization, switching between push-style and pull-style traversal based on frontier size and graph structure.

Key tuning parameters:

```bash
./build/bfs_graphfirst <graph.edgelist> <source> --alpha 14 --beta 24
```

### MPI Distributed BFS

The MPI implementation targets distributed-memory CPU execution on Perlmutter. It evaluates how graph partitioning impacts communication and scalability.

Built binaries:

```text
build/bfs_mpi_1d
build/bfs_mpi_2d
```

The MPI workflow includes correctness tests on small edgelist graphs and benchmark runs on larger synthetic and SNAP road network graphs.

---

## Build

```bash
make
```

Useful build targets:

```bash
make linalg       # build linear-algebraic CUDA BFS
make graphfirst   # build graph-first CUDA BFS
make clean        # remove build artifacts
```

---

## Validation and Benchmarking

### GPU validation

```bash
./run.sh validate
```

### GPU benchmarks

```bash
./run.sh bench
python3 scripts/benchmark.py --repeats 10 --groups medium large
```

### Single-graph runs

```bash
./build/bfs_linalg test-graphs/large_sparse.edgelist 0 bitmap
./build/bfs_graphfirst test-graphs/large_sparse.edgelist 0 --alpha 14 --beta 24
```

Both GPU binaries support `--dump` to print raw BFS depths for validation.

---

## MPI / Perlmutter Workflow

Build the project, then submit the MPI test jobs from a Perlmutter login node:

```bash
make
sbatch scripts/test_mpi_edgelist.slurm
bash scripts/download_snap.sh
sbatch scripts/test_mpi_snap.slurm
```

Expected MPI result locations:

```text
results/mpi_raw/
├── correctness.txt
├── snap_correctness.txt
├── timing_medium.csv
├── timing_large.csv
├── timing_snap.csv
├── edgelist_test_<jobid>.out/.err
└── snap_test_<jobid>.out/.err
```

SNAP inputs are downloaded into:

```text
snap-graphs/
```

## Current MPI Results (CPU, Perlmutter)

Edgelist suite job `52329335` completed successfully and produced:

- `results/mpi_raw/correctness.txt` (tiny graphs; 1D ranks \(1,2,4\), 2D ranks \(1,4\))
- `results/mpi_raw/timing_medium.csv`
- `results/mpi_raw/timing_large.csv`
- Slurm log: `results/mpi_raw/edgelist_test_52329335.out`

SNAP road network tests (PA/TX/CA) write to:

- `results/mpi_raw/snap_correctness.txt`
- `results/mpi_raw/timing_snap.csv`

---

## Current GPU Results

The results below use median runtime over 10 single-GPU A100 runs on Perlmutter. Timings are measured with CUDA events and exclude graph loading and printing.

Source files:

```text
results/benchmark_20260428_042111.csv
results/summary_20260428_042111.csv
```

| Graph | Best SpMV variant | Best SpMV ms | Graph-first ms | Graph-first / SpMV |
|---|---:|---:|---:|---:|
| `medium_chain` | bitmap | 18.733 | 32.918 | 1.76x |
| `medium_dense` | warpbitmap | 0.557 | 0.697 | 1.25x |
| `medium_sparse` | warp | 0.665 | 0.758 | 1.14x |
| `large_dense` | bitmap | 4.438 | 5.085 | 1.15x |
| `large_powerlaw` | warpbitmap | 2.325 | 6.264 | 2.69x |
| `large_sparse` | bitmap | 1.446 | 1.927 | 1.33x |

### Takeaway

Across these six synthetic inputs, the best linear-algebraic SpMV kernel outperforms the graph-first implementation. Graph-first is most competitive on random sparse and dense graphs, while the largest gap appears on the power-law graph, where queue construction and atomic frontier updates add overhead around hub-heavy frontiers.