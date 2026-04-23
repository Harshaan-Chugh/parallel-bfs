# Linear Algebraic BFS (SpMV)

In this implementation, I framed BFS as repeated Sparse Matrix-Vector Multiplication (SpMV) on the graph's adjacency matrix in CSR format. The idea is that each BFS level is basically computing `frontier_next = A * frontier_current` — every unvisited vertex checks if any of its neighbors are in the current frontier. I iterate level-by-level until there's nothing new to discover. I ended up writing several kernel variants that you can pick at runtime: `./build/bfs_linalg <graph.edgelist> <source> [baseline|warp|bitmap|pushpull|warpbitmap]`.

## The Baseline Strategy
The **`baseline`** kernel is the simplest approach: it assigns exactly **one thread per vertex** (i.e., one thread per row in the adjacency matrix). Each thread looks at its vertex, loops through all of that vertex's neighbors, and checks if any neighbor is currently in the frontier (stored as a dense array of integers). If so, it marks itself as discovered and stops. All the other optimizations below are built to fix the bottlenecks of this baseline!

## Optimization 1: Warp-Cooperative Row Processing

- The baseline kernel gives one thread to each vertex. The problem is that if one vertex has 5,000 neighbors and another has 3, the thread stuck on the big vertex holds up the entire warp while 31 other threads do nothing.
- To fix this, I made all 32 threads in a warp work together on the same row. Each lane checks every 32nd edge, and then I use `__ballot_sync()` to combine their results — so we know instantly if any lane found a frontier neighbor.
- This works really well on sparse graphs with skewed degrees (like power-law graphs) since the hub vertices get processed way faster.
- The downside is that it launches 32× more threads overall, so on dense graphs where every row is already short, the extra overhead actually makes it slower.

## Optimization 2: Bitmap Frontier Representation

- The frontier array stores a 0 or 1 per vertex, but uses a full `int` (4 bytes) for each one. That's super wasteful — we're moving 32× more data than we need to every kernel launch.
- I packed the frontier into a `uint32_t` bitmap instead, where each bit represents one vertex. To check vertex `j`, I just do `frontier_bm[j/32] & (1 << (j%32))`. Writing to the new frontier uses `atomicOr` on the bitmap word.
- This shines on dense graphs with big frontiers. For a 100K-vertex graph, the frontier goes from 400 KB down to ~12.5 KB, which is way more cache-friendly.
- The tradeoff is that the bit-twiddling and `atomicOr` contention hurt on sparse graphs where the frontier is small anyway.

## Optimization 3: Push-Pull Hybrid (Direction-Optimizing SpMV)

- Building off the first two optimizations, I noticed that both were still "pull" based and every unvisited vertex scans its entire neighbor list every level, even if the frontier only has 1 vertex in it. This wastes a ton of work in the early and late stages of BFS.
- I added a "push" kernel where only the vertices *in the frontier* do work, pushing themselves to their neighbors. Then, I set up the host to automatically switch between the push kernel and the pull kernel every level based on the frontier's size. If the frontier is small, we push. If it's huge, we pull.

## Optimization 4: Fusing Warp-Cooperative + Bitmap

- Since warp-cooperative (Opt 1) fixes load imbalance on sparse graphs, and bitmap (Opt 2) fixes bandwidth on dense graphs, I wanted to see if I could combine them to get the best of both worlds.
- I wrote a fused kernel `spmv_bfs_warp_bitmap` that uses all 32 threads in a warp to cooperatively scan a row, but instead of reading a dense `int` array, they read from the compressed `uint32_t` bitmap frontier.
- This gives us the load balancing for high-degree hub vertices while simultaneously slashing the memory bandwidth by 32×. It covers the weaknesses of both individual optimizations!

## Runtime Results

Here's how all the optimizations stack up. These runtimes (in milliseconds) were captured on a single NVIDIA A100 GPU on NERSC Perlmutter. They measure the pure BFS kernel execution time (excluding graph loading and printing).

### Medium Graphs (1K vertices)
| Graph | Avg Degree | Depth | Baseline | Warp | Bitmap | Push-Pull | Warp-Bitmap |
|-------|-----------|-------|----------|------|--------|-----------|-------------|
| medium_chain | 2 | 999 | 20.19 | 21.57 | 19.02 | 32.11 | 23.95 |
| medium_dense | 100 | 2 | 5.30 | 7.02 | 0.58 | 0.65 | **0.57** |
| medium_sparse | 6 | 6 | 2.47 | 0.61 | 0.63 | 0.79 | 0.69 |

### Large Graphs (100K vertices)
| Graph | Avg Degree | Depth | Baseline | Warp | Bitmap | Push-Pull | Warp-Bitmap |
|-------|-----------|-------|----------|------|--------|-----------|-------------|
| large_dense | 100 | 4 | 4.76 | 14.00 | 4.44 | 4.67 | **4.41** |
| large_powerlaw | 40 | 3 | 2.81 | 2.56 | 2.46 | 4.38 | **2.41** |
| large_sparse | 10 | 7 | 11.34 | 1.79 | 18.15 | 1.79 | **1.58** |

*Note: The chain graph is a worst-case scenario for all GPU BFS algorithms because it forces 999 sequential kernel launches with virtually no parallelism per level.*

## References

1. **Direction-Optimizing BFS (Push-Pull):**
   *Beamer, S., Asanović, K., & Patterson, D. (2012). "Direction-Optimizing Breadth-First Search." Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis (SC12).*
   > This paper introduced the idea of switching between top-down (push) and bottom-up (pull) traversals based on the size of the frontier, which forms the basis of Optimization 3.

2. **Warp-Cooperative Scanning & Bitmap Frontiers:**
   *Merrill, D., Garland, M., & Grimshaw, A. (2012). "Scalable GPU Graph Traversal." ACM SIGPLAN Notices (PPoPP '12).*
   > This paper is a staple in GPU BFS literature. While they focus on prefix-sums and queue management, the techniques for handling load imbalance (warp-cooperative execution) and frontier compression (bitmaps) were popularized by this era of GPU research.

3. **Linear Algebraic Graph Foundations (SpMV):**
   *Kepner, J., et al. (2016). "Mathematical Foundations of the GraphBLAS." IEEE High Performance Extreme Computing Conference (HPEC).*
   > Frames graph traversals entirely as linear algebraic operations (like $A \times x$), which is the core concept behind this entire directory's approach.
