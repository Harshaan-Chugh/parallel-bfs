/*
 * bfs_graphfirst.cu - Direction-Optimizing BFS (Scott Beamer style)
 *
 * Approach:
 *   Traditional graph traversal with two strategies:
 *   - Top-down: frontier vertices "push" to unvisited neighbors
 *   - Bottom-up: unvisited vertices "pull" by checking if any neighbor is in frontier
 *   Switches direction based on frontier size heuristic.
 *
 * Build: see root Makefile
 * Usage: ./build/bfs_graphfirst <graph.edgelist> <source_vertex> [--alpha A] [--beta B] [--dump]
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

typedef struct {
    int levels;
    int top_down_steps;
    int bottom_up_steps;
} BfsStats;

/* ============================================================
 * GPU Kernels
 * ============================================================ */

/*
 * Top-down BFS kernel (push-based).
 *
 * Each thread handles one frontier vertex and explores its neighbors.
 * Good when frontier is small.
 *
 * Each frontier vertex pushes to unvisited neighbors, using atomicCAS
 * on visited[] to avoid duplicate discoveries.
 */
__global__ void bfs_top_down_kernel(
    const int *row_ptr,
    const int *col_idx,
    const int *frontier,      /* current frontier (queue or bitmap) */
    int        frontier_size,
    int       *visited,
    int       *new_frontier,
    int       *new_frontier_size,
    unsigned  *new_frontier_bitmap,
    unsigned long long *new_frontier_edges,
    int       *depth,
    int        level,
    int        num_vertices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    int u = frontier[idx];
    if (u < 0 || u >= num_vertices) return;

    for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
        int v = col_idx[e];
        if (atomicCAS(&visited[v], 0, 1) == 0) {
            int pos = atomicAdd(new_frontier_size, 1);
            new_frontier[pos] = v;
            atomicOr(&new_frontier_bitmap[v >> 5], 1u << (v & 31));
            atomicAdd(new_frontier_edges,
                      (unsigned long long)(row_ptr[v + 1] - row_ptr[v]));
            depth[v] = level;
        }
    }
}

/*
 * Bottom-up BFS kernel (pull-based).
 *
 * Each thread handles one unvisited vertex and checks if any neighbor
 * is in the frontier. Good when frontier is large.
 *
 * Each unvisited vertex checks neighbors for frontier membership and
 * stops early after finding one parent.
 */
__global__ void bfs_bottom_up_kernel(
    const int *row_ptr,
    const int *col_idx,
    const unsigned *frontier_bitmap, /* bitmap representation of frontier */
    int       *visited,
    int       *new_frontier,
    int       *new_frontier_size,
    unsigned  *new_frontier_bitmap,
    unsigned long long *new_frontier_edges,
    int       *depth,
    int        level,
    int        num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (visited[v]) return;

    for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
        int u = col_idx[e];
        unsigned word = frontier_bitmap[u >> 5];
        unsigned bit  = 1u << (u & 31);

        if (word & bit) {
            if (atomicCAS(&visited[v], 0, 1) == 0) {
                int pos = atomicAdd(new_frontier_size, 1);
                new_frontier[pos] = v;
                atomicOr(&new_frontier_bitmap[v >> 5], 1u << (v & 31));
                atomicAdd(new_frontier_edges,
                          (unsigned long long)(row_ptr[v + 1] - row_ptr[v]));
                depth[v] = level;
            }
            break;
        }
    }
}

/* ============================================================
 * Host BFS driver
 * ============================================================ */

BfsStats bfs_graphfirst(const Graph *g, int source, int *h_depth,
                        float alpha, float beta) {
    int V = g->num_vertices;
    int E = g->num_edges;
    int bm_words = (V + 31) / 32;

    BfsStats stats;
    stats.levels = 0;
    stats.top_down_steps = 0;
    stats.bottom_up_steps = 0;

    /* Initialize host depth array */
    for (int i = 0; i < V; i++) h_depth[i] = -1;
    h_depth[source] = 0;

    /* ---- Allocate device memory ---- */
    int *d_row_ptr, *d_col_idx;
    int *d_frontier, *d_new_frontier;
    unsigned *d_frontier_bitmap, *d_new_frontier_bitmap;
    int *d_visited, *d_depth;
    int *d_new_frontier_size;
    unsigned long long *d_new_frontier_edges;

    CUDA_CHECK(cudaMalloc(&d_row_ptr,             (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,             E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier,            V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_frontier,        V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier_bitmap,     bm_words * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_new_frontier_bitmap, bm_words * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_visited,             V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_depth,               V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_frontier_size,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_frontier_edges,  sizeof(unsigned long long)));

    /* ---- Copy graph to device ---- */
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g->row_ptr, (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g->col_idx, E * sizeof(int),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_depth,   h_depth,    V * sizeof(int),       cudaMemcpyHostToDevice));

    /* ---- Initialize frontier queue, frontier bitmap, and visited ---- */
    int *h_frontier = (int *)calloc(V, sizeof(int));
    int *h_visited = (int *)calloc(V, sizeof(int));
    unsigned *h_frontier_bitmap = (unsigned *)calloc(bm_words, sizeof(unsigned));
    if (!h_frontier || !h_visited || !h_frontier_bitmap) {
        fprintf(stderr, "Error: host allocation failed\n");
        exit(EXIT_FAILURE);
    }

    h_frontier[0] = source;
    h_visited[source] = 1;
    h_frontier_bitmap[source >> 5] |= (1u << (source & 31));

    CUDA_CHECK(cudaMemcpy(d_frontier,        h_frontier,        V * sizeof(int),              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited,         h_visited,         V * sizeof(int),              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier_bitmap, h_frontier_bitmap, bm_words * sizeof(unsigned),  cudaMemcpyHostToDevice));

    free(h_frontier);
    free(h_visited);
    free(h_frontier_bitmap);

    /* ---- BFS iteration loop with Beamer-style direction switching ---- */
    int level = 1;
    int frontier_size = 1;
    unsigned long long frontier_edges =
        (unsigned long long)(g->row_ptr[source + 1] - g->row_ptr[source]);
    int use_bottom_up = 0;

    const int blockSize = 256;

    while (frontier_size > 0) {
        if (!use_bottom_up && ((double)frontier_edges > ((double)E / (double)alpha))) {
            use_bottom_up = 1;
        } else if (use_bottom_up && ((double)frontier_size < ((double)V / (double)beta))) {
            use_bottom_up = 0;
        }

        int zero = 0;
        unsigned long long zero_edges = 0;
        CUDA_CHECK(cudaMemcpy(d_new_frontier_size, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_new_frontier_edges, &zero_edges, sizeof(unsigned long long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_new_frontier_bitmap, 0, bm_words * sizeof(unsigned)));

        if (use_bottom_up) {
            int gridSize = (V + blockSize - 1) / blockSize;
            bfs_bottom_up_kernel<<<gridSize, blockSize>>>(
                d_row_ptr, d_col_idx,
                d_frontier_bitmap,
                d_visited,
                d_new_frontier,
                d_new_frontier_size,
                d_new_frontier_bitmap,
                d_new_frontier_edges,
                d_depth,
                level,
                V);
            stats.bottom_up_steps++;
        } else {
            int gridSize = (frontier_size + blockSize - 1) / blockSize;
            bfs_top_down_kernel<<<gridSize, blockSize>>>(
                d_row_ptr, d_col_idx,
                d_frontier,
                frontier_size,
                d_visited,
                d_new_frontier,
                d_new_frontier_size,
                d_new_frontier_bitmap,
                d_new_frontier_edges,
                d_depth,
                level,
                V);
            stats.top_down_steps++;
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&frontier_size, d_new_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&frontier_edges, d_new_frontier_edges, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        int *tmp_frontier = d_frontier;
        d_frontier = d_new_frontier;
        d_new_frontier = tmp_frontier;

        unsigned *tmp_bitmap = d_frontier_bitmap;
        d_frontier_bitmap = d_new_frontier_bitmap;
        d_new_frontier_bitmap = tmp_bitmap;

        level++;
    }

    stats.levels = (level >= 2) ? (level - 2) : 0;

    /* ---- Copy results back ---- */
    CUDA_CHECK(cudaMemcpy(h_depth, d_depth, V * sizeof(int), cudaMemcpyDeviceToHost));

    /* ---- Cleanup ---- */
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_new_frontier));
    CUDA_CHECK(cudaFree(d_frontier_bitmap));
    CUDA_CHECK(cudaFree(d_new_frontier_bitmap));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_new_frontier_size));
    CUDA_CHECK(cudaFree(d_new_frontier_edges));

    return stats;
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <graph.edgelist> <source_vertex> [--alpha A] [--beta B] [--dump]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const char *graph_file = argv[1];
    int source = atoi(argv[2]);
    float alpha = 14.0f;
    float beta = 24.0f;
    int dump = 0;

    for (int a = 3; a < argc; a++) {
        if (strcmp(argv[a], "--dump") == 0) {
            dump = 1;
        } else if (strcmp(argv[a], "--alpha") == 0 && a + 1 < argc) {
            alpha = (float)atof(argv[++a]);
        } else if (strcmp(argv[a], "--beta") == 0 && a + 1 < argc) {
            beta = (float)atof(argv[++a]);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[a]);
            return EXIT_FAILURE;
        }
    }

    if (alpha <= 0.0f || beta <= 0.0f) {
        fprintf(stderr, "Error: --alpha and --beta must be positive\n");
        return EXIT_FAILURE;
    }

    /* Use only 1 GPU (GPU 0) */
    CUDA_CHECK(cudaSetDevice(0));

    /* Load graph */
    Graph g;
    if (load_graph(graph_file, &g) != 0) {
        return EXIT_FAILURE;
    }
    print_graph_info(&g);

    if (source < 0 || source >= g.num_vertices) {
        fprintf(stderr, "Error: source vertex %d out of range [0, %d)\n",
                source, g.num_vertices);
        free_graph(&g);
        return EXIT_FAILURE;
    }

    printf("Algorithm: graph-first direction-optimizing BFS (alpha=%.1f, beta=%.1f)\n",
           alpha, beta);

    /* Run timed BFS */
    int *depth = (int *)malloc(g.num_vertices * sizeof(int));

    cudaEvent_t t_start, t_stop;
    CUDA_CHECK(cudaEventCreate(&t_start));
    CUDA_CHECK(cudaEventCreate(&t_stop));
    CUDA_CHECK(cudaEventRecord(t_start));

    BfsStats stats = bfs_graphfirst(&g, source, depth, alpha, beta);

    CUDA_CHECK(cudaEventRecord(t_stop));
    CUDA_CHECK(cudaEventSynchronize(t_stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop));
    printf("BFS time: %.3f ms\n", elapsed_ms);
    printf("Graph-first max level: %d | top-down launches: %d | bottom-up launches: %d\n",
           stats.levels, stats.top_down_steps, stats.bottom_up_steps);
    CUDA_CHECK(cudaEventDestroy(t_start));
    CUDA_CHECK(cudaEventDestroy(t_stop));

    /* Print results */
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
