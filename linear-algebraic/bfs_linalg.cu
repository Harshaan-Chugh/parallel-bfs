/*
 * bfs_linalg.cu - Linear Algebraic BFS via Sparse Matrix-Vector Multiply (SpMV)
 *
 * Approach:
 *   BFS is mapped to repeated SpMV on the adjacency matrix in CSR format.
 *   frontier_next = A * frontier_current  (masking visited vertices)
 *
 * The adjacency matrix A is stored in CSR (row_ptr, col_idx).
 * The frontier vectors are dense boolean/int vectors of length num_vertices.
 *
 * Kernel variants (selectable at runtime):
 *   baseline   - one thread per row, scalar scan
 *   warp       - warp-cooperative row processing for load balance
 *   bitmap     - bitmap-compressed frontier (32x memory reduction)
 *   pushpull   - direction-optimizing: push when frontier small, pull when large
 *   warpbitmap - fused warp-cooperative + bitmap frontier
 *
 * Build: see root Makefile
 * Usage: ./build/bfs_linalg <graph.edgelist> <source_vertex> [baseline|warp|bitmap|pushpull|warpbitmap] [--dump]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "../test-graphs/graph_loader.h"

/* ============================================================
 * CUDA error checking
 * ============================================================ */
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ============================================================
 * Kernel variant selection
 * ============================================================ */
enum KernelVariant { BASELINE = 0, WARP_COOP = 1, BITMAP = 2, PUSHPULL = 3, WARP_BITMAP = 4 };

/* ============================================================
 * Kernel 1: Baseline SpMV BFS (one thread per row)
 *
 * Each thread owns one row (vertex) of the adjacency matrix.
 * It checks whether any neighbor is in the current frontier.
 * This is the classic pull-based SpMV: y = A * x
 *   x = frontier, y = new_frontier
 * ============================================================ */
__global__ void spmv_bfs_baseline(
    const int *row_ptr,
    const int *col_idx,
    const int *frontier,
    int       *visited,
    int       *new_frontier,
    int       *depth,
    int        level,
    int        num_vertices,
    int       *found_new)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;
    if (visited[i]) return;  /* already discovered — skip entire row */

    int row_start = row_ptr[i];
    int row_end   = row_ptr[i + 1];

    for (int e = row_start; e < row_end; e++) {
        if (frontier[col_idx[e]]) {   /* neighbor in frontier? */
            new_frontier[i] = 1;
            visited[i]      = 1;
            depth[i]        = level;
            atomicOr(found_new, 1);   /* signal host: keep going */
            break;                    /* one parent suffices for BFS */
        }
    }
}

/* ============================================================
 * Kernel 2: Warp-Cooperative SpMV BFS
 *
 * Problem: high-degree vertices cause load imbalance — one thread
 * scans thousands of edges while 31 warp-mates idle.
 *
 * Solution: all 32 lanes in a warp cooperatively scan the same row.
 * Each lane processes every 32nd edge. __any_sync aggregates results.
 * ============================================================ */
__global__ void spmv_bfs_warp(
    const int *row_ptr,
    const int *col_idx,
    const int *frontier,
    int       *visited,
    int       *new_frontier,
    int       *depth,
    int        level,
    int        num_vertices,
    int       *found_new)
{
    /* Each warp processes one vertex */
    int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane      = threadIdx.x & 31;   /* lane within the warp (0-31) */

    if (warp_id >= num_vertices) return;

    int i = warp_id;
    if (visited[i]) return;

    int row_start = row_ptr[i];
    int row_end   = row_ptr[i + 1];
    int hit       = 0;

    /* Each lane strides through the row's edges */
    for (int e = row_start + lane; e < row_end; e += 32) {
        if (frontier[col_idx[e]]) {
            hit = 1;
            break;    /* this lane found a match */
        }
    }

    /* Aggregate across the warp: did ANY lane find a frontier neighbor? */
    unsigned mask = __ballot_sync(0xFFFFFFFF, hit);
    if (mask && lane == 0) {
        new_frontier[i] = 1;
        visited[i]      = 1;
        depth[i]        = level;
        atomicOr(found_new, 1);
    }
}

/* ============================================================
 * Kernel 3: Bitmap Frontier SpMV BFS
 *
 * Problem: dense int frontier vectors waste bandwidth — 4 bytes
 * per vertex but only 1 bit of information.
 *
 * Solution: pack the frontier into a uint32 bitmap.
 * Each bit represents one vertex. 32x memory reduction.
 * ============================================================ */
__global__ void spmv_bfs_bitmap(
    const int      *row_ptr,
    const int      *col_idx,
    const unsigned *frontier_bm,   /* bitmap: bit j set = vertex j in frontier */
    int            *visited,
    unsigned       *new_frontier_bm,
    int            *depth,
    int             level,
    int             num_vertices,
    int            *found_new)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;
    if (visited[i]) return;

    int row_start = row_ptr[i];
    int row_end   = row_ptr[i + 1];

    for (int e = row_start; e < row_end; e++) {
        int j     = col_idx[e];
        /* Check if bit j is set in the frontier bitmap */
        unsigned word = frontier_bm[j >> 5];        /* j / 32 */
        unsigned bit  = 1u << (j & 31);             /* j % 32 */
        if (word & bit) {
            /* Set bit i in new_frontier bitmap (atomic) */
            atomicOr(&new_frontier_bm[i >> 5], 1u << (i & 31));
            visited[i] = 1;
            depth[i]   = level;
            atomicOr(found_new, 1);
            break;
        }
    }
}

/* ============================================================
 * Kernel 4: Push kernel (frontier vertices expand to neighbors)
 *
 * Only vertices IN the frontier do work. Each frontier vertex
 * iterates its neighbors and marks unvisited ones as discovered.
 * Uses atomicCAS on visited[] to avoid duplicate discoveries.
 * Also counts the new frontier size for direction-switching.
 * ============================================================ */
__global__ void spmv_bfs_push(
    const int *row_ptr,
    const int *col_idx,
    const int *frontier,
    int       *visited,
    int       *new_frontier,
    int       *depth,
    int        level,
    int        num_vertices,
    int       *found_new,
    int       *new_frontier_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;
    if (!frontier[i]) return;  /* only frontier vertices do work */

    int row_start = row_ptr[i];
    int row_end   = row_ptr[i + 1];

    for (int e = row_start; e < row_end; e++) {
        int j = col_idx[e];
        /* atomicCAS: try to set visited[j] from 0 to 1 */
        if (atomicCAS(&visited[j], 0, 1) == 0) {
            new_frontier[j] = 1;
            depth[j]        = level;
            atomicOr(found_new, 1);
            atomicAdd(new_frontier_count, 1);
        }
    }
}

/* ============================================================
 * Kernel 5: Pull kernel with frontier counting
 *
 * Same as baseline pull, but also counts new frontier vertices
 * so the host can decide push vs pull for the next level.
 * ============================================================ */
__global__ void spmv_bfs_pull(
    const int *row_ptr,
    const int *col_idx,
    const int *frontier,
    int       *visited,
    int       *new_frontier,
    int       *depth,
    int        level,
    int        num_vertices,
    int       *found_new,
    int       *new_frontier_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;
    if (visited[i]) return;

    int row_start = row_ptr[i];
    int row_end   = row_ptr[i + 1];

    for (int e = row_start; e < row_end; e++) {
        if (frontier[col_idx[e]]) {
            new_frontier[i] = 1;
            visited[i]      = 1;
            depth[i]        = level;
            atomicOr(found_new, 1);
            atomicAdd(new_frontier_count, 1);
            break;
        }
    }
}

/* ============================================================
 * Kernel 6: Fused Warp-Cooperative + Bitmap Frontier
 *
 * Combines the best of both worlds:
 *   - Warp-cooperative row scanning for load balance (Opt 1)
 *   - Bitmap frontier for 32x bandwidth reduction (Opt 2)
 * ============================================================ */
__global__ void spmv_bfs_warp_bitmap(
    const int      *row_ptr,
    const int      *col_idx,
    const unsigned *frontier_bm,
    int            *visited,
    unsigned       *new_frontier_bm,
    int            *depth,
    int             level,
    int             num_vertices,
    int            *found_new)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x & 31;

    if (warp_id >= num_vertices) return;

    int i = warp_id;
    if (visited[i]) return;

    int row_start = row_ptr[i];
    int row_end   = row_ptr[i + 1];
    int hit       = 0;

    /* Each lane strides through edges, checking bitmap frontier */
    for (int e = row_start + lane; e < row_end; e += 32) {
        int j        = col_idx[e];
        unsigned word = frontier_bm[j >> 5];
        unsigned bit  = 1u << (j & 31);
        if (word & bit) {
            hit = 1;
            break;
        }
    }

    /* Aggregate across the warp */
    unsigned mask = __ballot_sync(0xFFFFFFFF, hit);
    if (mask && lane == 0) {
        atomicOr(&new_frontier_bm[i >> 5], 1u << (i & 31));
        visited[i] = 1;
        depth[i]   = level;
        atomicOr(found_new, 1);
    }
}

/* ============================================================
 * Host BFS driver — baseline & warp variants (dense int frontier)
 * ============================================================ */

void bfs_linalg_dense(const Graph *g, int source, int *h_depth, KernelVariant variant) {
    int V = g->num_vertices;
    int E = g->num_edges;

    /* Initialize host depth array */
    for (int i = 0; i < V; i++) h_depth[i] = -1;
    h_depth[source] = 0;

    /* ---- Allocate device memory ---- */
    int *d_row_ptr, *d_col_idx;
    int *d_frontier, *d_new_frontier;
    int *d_visited, *d_depth;
    int *d_found_new;

    CUDA_CHECK(cudaMalloc(&d_row_ptr,      (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,      E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier,     V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_frontier, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited,      V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_depth,        V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found_new,    sizeof(int)));

    /* ---- Copy graph to device ---- */
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g->row_ptr, (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g->col_idx, E * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth,   h_depth,    V * sizeof(int),       cudaMemcpyHostToDevice));

    /* ---- Initialize frontier and visited on device ---- */
    int *h_frontier = (int *)calloc(V, sizeof(int));
    int *h_visited  = (int *)calloc(V, sizeof(int));
    h_frontier[source] = 1;
    h_visited[source]  = 1;

    CUDA_CHECK(cudaMemcpy(d_frontier, h_frontier, V * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited,  h_visited,  V * sizeof(int), cudaMemcpyHostToDevice));

    free(h_frontier);
    free(h_visited);

    /* ---- BFS iteration ---- */
    int level = 1;
    int h_found_new = 1;

    int blockSize = 256;
    int gridSize;

    if (variant == WARP_COOP) {
        /* Warp variant: one warp (32 threads) per vertex */
        gridSize = (V * 32 + blockSize - 1) / blockSize;
    } else {
        /* Baseline: one thread per vertex */
        gridSize = (V + blockSize - 1) / blockSize;
    }

    while (h_found_new) {
        h_found_new = 0;
        CUDA_CHECK(cudaMemcpy(d_found_new, &h_found_new, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_new_frontier, 0, V * sizeof(int)));

        if (variant == WARP_COOP) {
            spmv_bfs_warp<<<gridSize, blockSize>>>(
                d_row_ptr, d_col_idx,
                d_frontier, d_visited, d_new_frontier,
                d_depth, level, V, d_found_new);
        } else {
            spmv_bfs_baseline<<<gridSize, blockSize>>>(
                d_row_ptr, d_col_idx,
                d_frontier, d_visited, d_new_frontier,
                d_depth, level, V, d_found_new);
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        /* Check if any new vertices were found */
        CUDA_CHECK(cudaMemcpy(&h_found_new, d_found_new, sizeof(int), cudaMemcpyDeviceToHost));

        /* Swap frontier pointers */
        int *tmp = d_frontier;
        d_frontier = d_new_frontier;
        d_new_frontier = tmp;

        level++;
    }

    /* ---- Copy results back ---- */
    CUDA_CHECK(cudaMemcpy(h_depth, d_depth, V * sizeof(int), cudaMemcpyDeviceToHost));

    /* ---- Cleanup ---- */
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_new_frontier));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_found_new));
}

/* ============================================================
 * Host BFS driver — bitmap variant
 * ============================================================ */

void bfs_linalg_bitmap(const Graph *g, int source, int *h_depth) {
    int V = g->num_vertices;
    int E = g->num_edges;
    int bm_words = (V + 31) / 32;  /* number of uint32 words in bitmap */

    /* Initialize host depth array */
    for (int i = 0; i < V; i++) h_depth[i] = -1;
    h_depth[source] = 0;

    /* ---- Allocate device memory ---- */
    int      *d_row_ptr, *d_col_idx;
    unsigned *d_frontier_bm, *d_new_frontier_bm;
    int      *d_visited, *d_depth;
    int      *d_found_new;

    CUDA_CHECK(cudaMalloc(&d_row_ptr,         (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,         E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier_bm,     bm_words * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_new_frontier_bm, bm_words * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_visited,         V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_depth,           V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found_new,       sizeof(int)));

    /* ---- Copy graph to device ---- */
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g->row_ptr, (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g->col_idx, E * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth,   h_depth,    V * sizeof(int),       cudaMemcpyHostToDevice));

    /* ---- Initialize frontier bitmap and visited ---- */
    unsigned *h_frontier_bm = (unsigned *)calloc(bm_words, sizeof(unsigned));
    int      *h_visited     = (int *)calloc(V, sizeof(int));
    h_frontier_bm[source >> 5] |= (1u << (source & 31));
    h_visited[source] = 1;

    CUDA_CHECK(cudaMemcpy(d_frontier_bm, h_frontier_bm, bm_words * sizeof(unsigned), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited,     h_visited,     V * sizeof(int),              cudaMemcpyHostToDevice));

    free(h_frontier_bm);
    free(h_visited);

    /* ---- BFS iteration ---- */
    int level = 1;
    int h_found_new = 1;
    int blockSize = 256;
    int gridSize  = (V + blockSize - 1) / blockSize;

    while (h_found_new) {
        h_found_new = 0;
        CUDA_CHECK(cudaMemcpy(d_found_new, &h_found_new, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_new_frontier_bm, 0, bm_words * sizeof(unsigned)));

        spmv_bfs_bitmap<<<gridSize, blockSize>>>(
            d_row_ptr, d_col_idx,
            d_frontier_bm, d_visited, d_new_frontier_bm,
            d_depth, level, V, d_found_new);

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_found_new, d_found_new, sizeof(int), cudaMemcpyDeviceToHost));

        /* Swap frontier bitmap pointers */
        unsigned *tmp = d_frontier_bm;
        d_frontier_bm = d_new_frontier_bm;
        d_new_frontier_bm = tmp;

        level++;
    }

    /* ---- Copy results back ---- */
    CUDA_CHECK(cudaMemcpy(h_depth, d_depth, V * sizeof(int), cudaMemcpyDeviceToHost));

    /* ---- Cleanup ---- */
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_frontier_bm));
    CUDA_CHECK(cudaFree(d_new_frontier_bm));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_found_new));
}

/* ============================================================
 * Host BFS driver — push-pull hybrid (direction-optimizing)
 *
 * Switches between push and pull each level based on frontier size.
 * Beamer's heuristic: pull when edges_from_frontier > E / alpha.
 * ============================================================ */

void bfs_linalg_pushpull(const Graph *g, int source, int *h_depth) {
    int V = g->num_vertices;
    int E = g->num_edges;
    float alpha = 14.0f;  /* Beamer's switching threshold */

    /* Initialize host depth array */
    for (int i = 0; i < V; i++) h_depth[i] = -1;
    h_depth[source] = 0;

    /* ---- Allocate device memory ---- */
    int *d_row_ptr, *d_col_idx;
    int *d_frontier, *d_new_frontier;
    int *d_visited, *d_depth;
    int *d_found_new, *d_new_count;

    CUDA_CHECK(cudaMalloc(&d_row_ptr,      (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,      E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier,     V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_frontier, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited,      V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_depth,        V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found_new,    sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_count,    sizeof(int)));

    /* ---- Copy graph to device ---- */
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g->row_ptr, (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g->col_idx, E * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth,   h_depth,    V * sizeof(int),       cudaMemcpyHostToDevice));

    /* ---- Initialize frontier and visited ---- */
    int *h_frontier = (int *)calloc(V, sizeof(int));
    int *h_visited  = (int *)calloc(V, sizeof(int));
    h_frontier[source] = 1;
    h_visited[source]  = 1;

    CUDA_CHECK(cudaMemcpy(d_frontier, h_frontier, V * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited,  h_visited,  V * sizeof(int), cudaMemcpyHostToDevice));

    free(h_frontier);
    free(h_visited);

    /* ---- BFS iteration ---- */
    int level = 1;
    int h_found_new = 1;
    int frontier_count = 1;  /* source vertex */
    int avg_degree = (V > 0) ? E / V : 1;

    int blockSize = 256;
    int gridSize  = (V + blockSize - 1) / blockSize;

    while (h_found_new) {
        h_found_new = 0;
        int h_new_count = 0;
        CUDA_CHECK(cudaMemcpy(d_found_new, &h_found_new, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_new_count, &h_new_count, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_new_frontier, 0, V * sizeof(int)));

        /* Beamer's heuristic: estimate edges to check from frontier */
        long long edges_from_frontier = (long long)frontier_count * avg_degree;
        int use_pull = (edges_from_frontier > (long long)(E / alpha));

        if (use_pull) {
            spmv_bfs_pull<<<gridSize, blockSize>>>(
                d_row_ptr, d_col_idx,
                d_frontier, d_visited, d_new_frontier,
                d_depth, level, V, d_found_new, d_new_count);
        } else {
            spmv_bfs_push<<<gridSize, blockSize>>>(
                d_row_ptr, d_col_idx,
                d_frontier, d_visited, d_new_frontier,
                d_depth, level, V, d_found_new, d_new_count);
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_found_new, d_found_new, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&frontier_count, d_new_count, sizeof(int), cudaMemcpyDeviceToHost));

        /* Swap frontier pointers */
        int *tmp = d_frontier;
        d_frontier = d_new_frontier;
        d_new_frontier = tmp;

        level++;
    }

    /* ---- Copy results back ---- */
    CUDA_CHECK(cudaMemcpy(h_depth, d_depth, V * sizeof(int), cudaMemcpyDeviceToHost));

    /* ---- Cleanup ---- */
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_new_frontier));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_found_new));
    CUDA_CHECK(cudaFree(d_new_count));
}

/* ============================================================
 * Host BFS driver — fused warp+bitmap variant
 * ============================================================ */

void bfs_linalg_warp_bitmap(const Graph *g, int source, int *h_depth) {
    int V = g->num_vertices;
    int E = g->num_edges;
    int bm_words = (V + 31) / 32;

    /* Initialize host depth array */
    for (int i = 0; i < V; i++) h_depth[i] = -1;
    h_depth[source] = 0;

    /* ---- Allocate device memory ---- */
    int      *d_row_ptr, *d_col_idx;
    unsigned *d_frontier_bm, *d_new_frontier_bm;
    int      *d_visited, *d_depth;
    int      *d_found_new;

    CUDA_CHECK(cudaMalloc(&d_row_ptr,         (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,         E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier_bm,     bm_words * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_new_frontier_bm, bm_words * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_visited,         V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_depth,           V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found_new,       sizeof(int)));

    /* ---- Copy graph to device ---- */
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g->row_ptr, (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g->col_idx, E * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth,   h_depth,    V * sizeof(int),       cudaMemcpyHostToDevice));

    /* ---- Initialize frontier bitmap and visited ---- */
    unsigned *h_frontier_bm = (unsigned *)calloc(bm_words, sizeof(unsigned));
    int      *h_visited     = (int *)calloc(V, sizeof(int));
    h_frontier_bm[source >> 5] |= (1u << (source & 31));
    h_visited[source] = 1;

    CUDA_CHECK(cudaMemcpy(d_frontier_bm, h_frontier_bm, bm_words * sizeof(unsigned), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited,     h_visited,     V * sizeof(int),              cudaMemcpyHostToDevice));

    free(h_frontier_bm);
    free(h_visited);

    /* ---- BFS iteration ---- */
    int level = 1;
    int h_found_new = 1;
    int blockSize = 256;
    /* Warp variant: one warp (32 threads) per vertex */
    int gridSize = (V * 32 + blockSize - 1) / blockSize;

    while (h_found_new) {
        h_found_new = 0;
        CUDA_CHECK(cudaMemcpy(d_found_new, &h_found_new, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_new_frontier_bm, 0, bm_words * sizeof(unsigned)));

        spmv_bfs_warp_bitmap<<<gridSize, blockSize>>>(
            d_row_ptr, d_col_idx,
            d_frontier_bm, d_visited, d_new_frontier_bm,
            d_depth, level, V, d_found_new);

        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_found_new, d_found_new, sizeof(int), cudaMemcpyDeviceToHost));

        /* Swap frontier bitmap pointers */
        unsigned *tmp = d_frontier_bm;
        d_frontier_bm = d_new_frontier_bm;
        d_new_frontier_bm = tmp;

        level++;
    }

    /* ---- Copy results back ---- */
    CUDA_CHECK(cudaMemcpy(h_depth, d_depth, V * sizeof(int), cudaMemcpyDeviceToHost));

    /* ---- Cleanup ---- */
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_frontier_bm));
    CUDA_CHECK(cudaFree(d_new_frontier_bm));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_found_new));
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <graph.edgelist> <source_vertex> [baseline|warp|bitmap|pushpull|warpbitmap] [--dump]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const char *graph_file = argv[1];
    int source = atoi(argv[2]);

    /* Parse optional kernel variant */
    KernelVariant variant = BASELINE;
    int dump = 0;

    for (int a = 3; a < argc; a++) {
        if (strcmp(argv[a], "warp") == 0)            variant = WARP_COOP;
        else if (strcmp(argv[a], "bitmap") == 0)      variant = BITMAP;
        else if (strcmp(argv[a], "baseline") == 0)     variant = BASELINE;
        else if (strcmp(argv[a], "pushpull") == 0)     variant = PUSHPULL;
        else if (strcmp(argv[a], "warpbitmap") == 0)   variant = WARP_BITMAP;
        else if (strcmp(argv[a], "--dump") == 0)       dump = 1;
    }

    const char *variant_names[] = {"baseline", "warp-cooperative", "bitmap", "push-pull", "warp-bitmap"};

    /* Use only 1 GPU (GPU 0) */
    CUDA_CHECK(cudaSetDevice(0));

    /* Load graph */
    Graph g;
    if (load_graph(graph_file, &g) != 0) {
        return EXIT_FAILURE;
    }
    print_graph_info(&g);
    printf("Kernel variant: %s\n", variant_names[variant]);

    if (source < 0 || source >= g.num_vertices) {
        fprintf(stderr, "Error: source vertex %d out of range [0, %d)\n",
                source, g.num_vertices);
        free_graph(&g);
        return EXIT_FAILURE;
    }

    /* Allocate depth array */
    int *depth = (int *)malloc(g.num_vertices * sizeof(int));

    /* ---- Timed BFS ---- */
    cudaEvent_t t_start, t_stop;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_stop));
    CUDA_CHECK(cudaEventRecord(t_start));

    if (variant == BITMAP) {
        bfs_linalg_bitmap(&g, source, depth);
    } else if (variant == PUSHPULL) {
        bfs_linalg_pushpull(&g, source, depth);
    } else if (variant == WARP_BITMAP) {
        bfs_linalg_warp_bitmap(&g, source, depth);
    } else {
        bfs_linalg_dense(&g, source, depth, variant);
    }

    CUDA_CHECK(cudaEventRecord(t_stop));
    CUDA_CHECK(cudaEventSynchronize(t_stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop));
    printf("BFS time: %.3f ms\n", elapsed_ms);
    CUDA_CHECK(cudaEventDestroy(t_start));
    CUDA_CHECK(cudaEventDestroy(t_stop));

    /* ---- Print results ---- */
    if (g.num_vertices <= 50) {
        printf("BFS depths from source %d:\n", source);
        for (int i = 0; i < g.num_vertices; i++) {
            printf("  vertex %d: depth %d\n", i, depth[i]);
        }
    } else {
        int reachable = 0, max_depth = 0;
        for (int i = 0; i < g.num_vertices; i++) {
            if (depth[i] >= 0) reachable++;
            if (depth[i] > max_depth) max_depth = depth[i];
        }
        printf("Source: %d | Reachable: %d/%d | Max depth: %d\n",
               source, reachable, g.num_vertices, max_depth);
    }

    /* ---- Dump raw depths for validation ---- */
    if (dump) {
        printf("--- DEPTHS ---\n");
        for (int i = 0; i < g.num_vertices; i++) {
            printf("%d\n", depth[i]);
        }
    }

    free(depth);
    free_graph(&g);
    return EXIT_SUCCESS;
}
