# Linear Algebraic BFS (SpMV)

In this implementation, I framed BFS as repeated Sparse Matrix-Vector Multiplication (SpMV) on the graph's adjacency matrix in CSR format. The idea is that each BFS level is basically computing `frontier_next = A * frontier_current` — every unvisited vertex checks if any of its neighbors are in the current frontier. I iterate level-by-level until there's nothing new to discover. I ended up writing three kernel variants (baseline, warp-cooperative, and bitmap) that you can pick at runtime: `./build/bfs_linalg <graph.edgelist> <source> [baseline|warp|bitmap]`.

## Optimization 1: Warp-Cooperative Row Processing

- The baseline kernel gives one thread to each vertex. The problem is that if one vertex has 5,000 neighbors and another has 3, the thread stuck on the big vertex holds up the entire warp while 31 other threads do nothing.
- To fix this, I made all 32 threads in a warp work together on the same row. Each lane checks every 32nd edge, and then I use `__ballot_sync()` to combine their results — so we know instantly if any lane found a frontier neighbor.
- This works really well on sparse graphs with skewed degrees (like power-law graphs) since the hub vertices get processed way faster.
- The downside is that it launches 32× more threads overall, so on dense graphs where every row is already short, the extra overhead actually makes it slower.

## Optimization 2: Bitmap Frontier Representation

- The frontier array stores a 0 or 1 per vertex, but uses a full `int` (4 bytes) for each one. That's super wasteful — we're moving 32× more data than we need to every kernel launch.
- I packed the frontier into a `uint32_t` bitmap instead, where each bit represents one vertex. To check vertex `j`, I just do `frontier_bm[j/32] & (1 << (j%32))`. Writing to the new frontier uses `atomicOr` on the bitmap word.
- This shines on dense graphs with big frontiers. For a 100K-vertex graph, the frontier goes from 400 KB down to ~12.5 KB, which is way more cache-friendly.
- The tradeoff is that the bit-twiddling and `atomicOr` contention hurt on sparse graphs where the frontier is small anyway — the overhead isn't worth the bandwidth savings.
