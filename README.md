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

More docs:

- **Linear-algebraic GPU BFS**: `linear-algebraic/README.md`
- **Graph-first GPU BFS**: `graph-first/README.md`
- **MPI BFS**: `mpi/README.md`

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
├── timing_medium_trials.csv
├── timing_medium_median.csv
├── timing_large_trials.csv
├── timing_large_median.csv
├── timing_snap.csv
├── edgelist_test_<jobid>.out/.err
└── snap_test_<jobid>.out/.err
```

SNAP inputs are downloaded into:

```text
snap-graphs/
```

## Current MPI Results (CPU, Perlmutter)

All raw MPI logs and CSVs live under `results/mpi_raw/`.

### Correctness

- **Edgelist tiny graphs**: all PASS (see `results/mpi_raw/correctness.txt`).
- **SNAP road networks**: for PA/TX/CA, the depth vector matches between **1D np=1 vs np=4**, and **1D vs 2D** (see `results/mpi_raw/snap_correctness.txt`).

### Timing: synthetic edgelists (median of 10 trials)

Source files:

```text
results/mpi_raw/timing_medium_trials.csv
results/mpi_raw/timing_medium_median.csv
results/mpi_raw/timing_large_trials.csv
results/mpi_raw/timing_large_median.csv
```

Jobs:

- Edgelist 10-trial timing job: `52331126` (log: `results/mpi_raw/edgelist_test_52331126.out`)

<!-- BEGIN AUTO-GENERATED: MPI edgelist medium (median of 10 trials) -->
| Graph | Strategy | Ranks | Median time (s) | Trials | Reachable | Max depth |
|---|---:|---:|---:|---:|---:|---:|
| `medium_chain` | 1D | 1 | 0.002195 | 10 | 1000 | 999 |
| `medium_chain` | 1D | 2 | 0.002547 | 10 | 1000 | 999 |
| `medium_chain` | 1D | 4 | 0.002883 | 10 | 1000 | 999 |
| `medium_chain` | 1D | 8 | 0.003789 | 10 | 1000 | 999 |
| `medium_chain` | 1D | 16 | 0.003820 | 10 | 1000 | 999 |
| `medium_chain` | 2D | 1 | 0.004441 | 10 | 1000 | 999 |
| `medium_chain` | 2D | 4 | 0.006936 | 10 | 1000 | 999 |
| `medium_chain` | 2D | 9 | 0.007510 | 10 | 1000 | 999 |
| `medium_chain` | 2D | 16 | 0.008404 | 10 | 1000 | 999 |
| `medium_dense` | 1D | 1 | 0.000123 | 10 | 1000 | 2 |
| `medium_dense` | 1D | 2 | 0.000073 | 10 | 1000 | 2 |
| `medium_dense` | 1D | 4 | 0.000050 | 10 | 1000 | 2 |
| `medium_dense` | 1D | 8 | 0.000052 | 10 | 1000 | 2 |
| `medium_dense` | 1D | 16 | 0.000034 | 10 | 1000 | 2 |
| `medium_dense` | 2D | 1 | 0.000124 | 10 | 1000 | 2 |
| `medium_dense` | 2D | 4 | 0.001151 | 10 | 1000 | 2 |
| `medium_dense` | 2D | 9 | 0.001169 | 10 | 1000 | 2 |
| `medium_dense` | 2D | 16 | 0.001171 | 10 | 1000 | 2 |
| `medium_sparse` | 1D | 1 | 0.000077 | 10 | 998 | 6 |
| `medium_sparse` | 1D | 2 | 0.000060 | 10 | 998 | 6 |
| `medium_sparse` | 1D | 4 | 0.000046 | 10 | 998 | 6 |
| `medium_sparse` | 1D | 8 | 0.000046 | 10 | 998 | 6 |
| `medium_sparse` | 1D | 16 | 0.000066 | 10 | 998 | 6 |
| `medium_sparse` | 2D | 1 | 0.000104 | 10 | 998 | 6 |
| `medium_sparse` | 2D | 4 | 0.001226 | 10 | 998 | 6 |
| `medium_sparse` | 2D | 9 | 0.001240 | 10 | 998 | 6 |
| `medium_sparse` | 2D | 16 | 0.001281 | 10 | 998 | 6 |
<!-- END AUTO-GENERATED: MPI edgelist medium (median of 10 trials) -->

<!-- BEGIN AUTO-GENERATED: MPI edgelist large (median of 10 trials) -->
| Graph | Strategy | Ranks | Median time (s) | Trials | Reachable | Max depth |
|---|---:|---:|---:|---:|---:|---:|
| `large_dense` | 1D | 1 | 0.021069 | 10 | 100000 | 4 |
| `large_dense` | 1D | 2 | 0.010233 | 10 | 100000 | 4 |
| `large_dense` | 1D | 4 | 0.005952 | 10 | 100000 | 4 |
| `large_dense` | 1D | 8 | 0.005335 | 10 | 100000 | 4 |
| `large_dense` | 1D | 16 | 0.006686 | 10 | 100000 | 4 |
| `large_dense` | 2D | 1 | 0.021171 | 10 | 100000 | 4 |
| `large_dense` | 2D | 4 | 0.006842 | 10 | 100000 | 4 |
| `large_dense` | 2D | 9 | 0.004279 | 10 | 100000 | 4 |
| `large_dense` | 2D | 16 | 0.003220 | 10 | 100000 | 4 |
| `large_powerlaw` | 1D | 1 | 0.005864 | 10 | 100000 | 3 |
| `large_powerlaw` | 1D | 2 | 0.003843 | 10 | 100000 | 3 |
| `large_powerlaw` | 1D | 4 | 0.002577 | 10 | 100000 | 3 |
| `large_powerlaw` | 1D | 8 | 0.001827 | 10 | 100000 | 3 |
| `large_powerlaw` | 1D | 16 | 0.001485 | 10 | 100000 | 3 |
| `large_powerlaw` | 2D | 1 | 0.006811 | 10 | 100000 | 3 |
| `large_powerlaw` | 2D | 4 | 0.004050 | 10 | 100000 | 3 |
| `large_powerlaw` | 2D | 9 | 0.003115 | 10 | 100000 | 3 |
| `large_powerlaw` | 2D | 16 | 0.002615 | 10 | 100000 | 3 |
| `large_sparse` | 1D | 1 | 0.008906 | 10 | 99996 | 7 |
| `large_sparse` | 1D | 2 | 0.005700 | 10 | 99996 | 7 |
| `large_sparse` | 1D | 4 | 0.004100 | 10 | 99996 | 7 |
| `large_sparse` | 1D | 8 | 0.003052 | 10 | 99996 | 7 |
| `large_sparse` | 1D | 16 | 0.003157 | 10 | 99996 | 7 |
| `large_sparse` | 2D | 1 | 0.011366 | 10 | 99996 | 7 |
| `large_sparse` | 2D | 4 | 0.005213 | 10 | 99996 | 7 |
| `large_sparse` | 2D | 9 | 0.004056 | 10 | 99996 | 7 |
| `large_sparse` | 2D | 16 | 0.003262 | 10 | 99996 | 7 |
<!-- END AUTO-GENERATED: MPI edgelist large (median of 10 trials) -->

### Timing: SNAP road networks (median of 3 trials)

Source file:

```text
results/mpi_raw/timing_snap.csv
```

Job:

- SNAP benchmark job: `52331036` (log: `results/mpi_raw/snap_test_52331036.out`)

<!-- BEGIN AUTO-GENERATED: MPI SNAP (median of 3 trials) -->
| Graph | Strategy | Ranks | Median time (s) | Trials |
|---|---:|---:|---:|---:|
| `roadNet-CA` | 1D | 1 | 5.532491 | 3 |
| `roadNet-CA` | 1D | 2 | 2.940258 | 3 |
| `roadNet-CA` | 1D | 4 | 1.844127 | 3 |
| `roadNet-CA` | 1D | 8 | 1.122868 | 3 |
| `roadNet-CA` | 1D | 16 | 0.714548 | 3 |
| `roadNet-CA` | 1D | 32 | 0.470961 | 3 |
| `roadNet-CA` | 1D | 64 | 0.412492 | 3 |
| `roadNet-CA` | 2D | 1 | 7.824045 | 3 |
| `roadNet-CA` | 2D | 4 | 4.121734 | 3 |
| `roadNet-CA` | 2D | 9 | 3.011879 | 3 |
| `roadNet-CA` | 2D | 16 | 2.555635 | 3 |
| `roadNet-CA` | 2D | 25 | 2.074113 | 3 |
| `roadNet-CA` | 2D | 36 | 1.896778 | 3 |
| `roadNet-CA` | 2D | 49 | 1.754454 | 3 |
| `roadNet-CA` | 2D | 64 | 1.611111 | 3 |
| `roadNet-PA` | 1D | 1 | 3.203230 | 3 |
| `roadNet-PA` | 1D | 2 | 1.679436 | 3 |
| `roadNet-PA` | 1D | 4 | 0.935017 | 3 |
| `roadNet-PA` | 1D | 8 | 0.566721 | 3 |
| `roadNet-PA` | 1D | 16 | 0.369628 | 3 |
| `roadNet-PA` | 1D | 32 | 0.252237 | 3 |
| `roadNet-PA` | 1D | 64 | 0.243842 | 3 |
| `roadNet-PA` | 2D | 1 | 4.431716 | 3 |
| `roadNet-PA` | 2D | 4 | 2.316969 | 3 |
| `roadNet-PA` | 2D | 9 | 1.728164 | 3 |
| `roadNet-PA` | 2D | 16 | 1.354890 | 3 |
| `roadNet-PA` | 2D | 25 | 1.156486 | 3 |
| `roadNet-PA` | 2D | 36 | 1.035597 | 3 |
| `roadNet-PA` | 2D | 49 | 0.966735 | 3 |
| `roadNet-PA` | 2D | 64 | 0.863500 | 3 |
| `roadNet-TX` | 1D | 1 | 5.392899 | 3 |
| `roadNet-TX` | 1D | 2 | 2.829177 | 3 |
| `roadNet-TX` | 1D | 4 | 1.546556 | 3 |
| `roadNet-TX` | 1D | 8 | 0.950362 | 3 |
| `roadNet-TX` | 1D | 16 | 0.621578 | 3 |
| `roadNet-TX` | 1D | 32 | 0.411471 | 3 |
| `roadNet-TX` | 1D | 64 | 0.383658 | 3 |
| `roadNet-TX` | 2D | 1 | 7.479369 | 3 |
| `roadNet-TX` | 2D | 4 | 3.922785 | 3 |
| `roadNet-TX` | 2D | 9 | 2.853090 | 3 |
| `roadNet-TX` | 2D | 16 | 2.219828 | 3 |
| `roadNet-TX` | 2D | 25 | 1.897720 | 3 |
| `roadNet-TX` | 2D | 36 | 1.688000 | 3 |
| `roadNet-TX` | 2D | 49 | 1.565437 | 3 |
| `roadNet-TX` | 2D | 64 | 1.390648 | 3 |
<!-- END AUTO-GENERATED: MPI SNAP (median of 3 trials) -->

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