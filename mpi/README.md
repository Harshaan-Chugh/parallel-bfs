# MPI-Distributed BFS

This directory has two pure-MPI (CPU-only, no CUDA) BFS implementations that partition the graph across processes using different strategies. Both are built from `graph_dist.h`, which handles loading — supporting the project's `.edgelist` format (header line `V E`, then `src dst` pairs) and the SNAP `.txt` format (`# comments`, tab-separated `src\tdst`, vertex count inferred from max ID). Run them with:

```
mpirun -np P ./build/bfs_mpi_1d <graph> <source> [--dump]
mpirun -np P ./build/bfs_mpi_2d <graph> <source> [--dump]   # P must be a perfect square
```

`--dump` prints `--- DEPTHS ---` followed by one depth per line (compatible with `test-graphs/validate_bfs.py`). Build with `make mpi` from the project root.

## Strategy 1: 1D Vertex Partitioning (`bfs_mpi_1d.c`)

Each rank owns a contiguous block of vertices (rows of the adjacency matrix). Per BFS level, every rank expands its local unvisited vertices by checking the global frontier, then contributes its newly discovered bits back. The global frontier is merged across all ranks via `MPI_Allreduce` with `MPI_BOR` on a dense bitmap — a single fixed-size collective per level. The depth array is merged at the end with `MPI_Allreduce(MPI_MAX)` so rank 0 can report complete results.

## Strategy 2: 2D Edge Partitioning (`bfs_mpi_2d.c`)

Arranges P ranks in a √P × √P process grid. The adjacency matrix is split into √P × √P submatrix blocks; rank (r, c) owns the block at row-block r, column-block c. Per BFS level: each rank extracts the frontier slice for its column block, scans its local submatrix to find newly reachable rows, then OR-reduces results across its process row with `MPI_Allreduce`. Row and column sub-communicators (from `MPI_Comm_split`) keep each collective to √P participants instead of all P, which is the key scalability advantage over 1D at high rank counts.

## Optimization 1: Bitmap Frontier

The frontier is stored as a dense 64-bit-word bitmap (one bit per vertex) instead of a list of vertex IDs. For roadNet-CA (~2M vertices) this is ~256 KB — fixed, compact, and directly usable as the buffer for `MPI_Allreduce(BOR)`. No dynamic sizing, no deduplication step.

## Optimization 2: Bitwise OR Reduction

`MPI_Allreduce` with `MPI_BOR` computes the union of all ranks' frontier contributions in a single collective call. This maps perfectly to set-union on bitmaps, and MPI's internal tree-reduction uses it efficiently. The alternative — gathering variable-length vertex lists — would require `Allgatherv` plus a deduplication pass.

## Optimization 3: Early Termination

After each `MPI_Allreduce(BOR)`, we call `popcount` on the result. If it's zero, the global frontier is empty and BFS terminates immediately — no extra collective needed. This avoids one wasted round per disconnected component.

## Optimization 4: Direction-Optimizing Pull

Each rank scans its locally owned unvisited vertices and checks if any neighbor is in the global frontier (bottom-up / pull). On undirected graphs this is always correct across partition boundaries: if edge (u, v) exists and u is in the frontier, then v's local adjacency list also contains u. This means each rank can discover all of its new vertices independently, without any push-style message exchange for remote neighbors.

## Optimization 5: Sub-Communicator Collectives (2D only)

The 2D BFS creates per-row and per-column sub-communicators via `MPI_Comm_split`. The row `Allreduce` and column operations run on groups of √P ranks instead of all P. At P = 64, this replaces O(log 64) steps touching 64 ranks with O(log 8) steps on 8 ranks, reducing bandwidth contention on multi-node runs.

## References

1. **Direction-Optimizing BFS:**
   *Beamer, S., Asanović, K., & Patterson, D. (2012). "Direction-Optimizing Breadth-First Search." SC12.*
   > Introduced push/pull switching based on frontier size — the basis for Optimization 4.

2. **2D Graph Partitioning:**
   *Buluç, A. & Gilbert, J. (2011). "The Combinatorial BLAS: Design, Implementation, and Applications." IJHPCA.*
   > Showed that 2D sparse matrix partitioning reduces communication to O(√P) per rank vs O(P) for 1D.

3. **SNAP Road Network Datasets:**
   *Leskovec, J., et al. (2009). "Community Structure in Large Networks." Internet Mathematics 6(1).*
   > Source of the roadNet-PA/TX/CA graphs used for benchmarking. Download with `bash scripts/download_snap.sh`.
