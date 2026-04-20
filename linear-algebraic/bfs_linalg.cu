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
 * Build: see root Makefile
 * Usage: ./build/bfs_linalg <graph.edgelist> <source_vertex>
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
 * GPU Kernels (TODO: implement these)
 * ============================================================ */

/*
 * SpMV-based frontier expansion kernel.
 *
 * For each vertex i (row of the adjacency matrix):
 *   If vertex i is NOT visited, check if any neighbor j has frontier[j] == 1.
 *   This is equivalent to: new_frontier[i] = OR over j in neighbors(i) of frontier[j]
 *   ...but only for unvisited vertices.
 *
 * Params:
 *   row_ptr   - CSR row pointers (size: num_vertices + 1)
 *   col_idx   - CSR column indices (size: num_edges)
 *   frontier  - current frontier vector (1 = in frontier, 0 = not)
 *   visited   - visited vector (1 = visited, 0 = not)
 *   new_frontier - output: next frontier
 *   depth     - output: BFS depth array
 *   level     - current BFS level
 *   num_vertices - number of vertices
 *   found_new - output: flag set to 1 if any new vertex was found
 */
__global__ void spmv_bfs_kernel(
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
    /* TODO: Implement SpMV-based BFS expansion
     *
     * Each thread handles one row (vertex) of the matrix:
     *   1. if visited[i] -> skip
     *   2. for each neighbor j in col_idx[row_ptr[i]..row_ptr[i+1]):
     *        if frontier[j] == 1:
     *          new_frontier[i] = 1
     *          visited[i] = 1
     *          depth[i] = level
     *          *found_new = 1
     *          break
     */
}

/* ============================================================
 * Host BFS driver
 * ============================================================ */

void bfs_linalg(const Graph *g, int source, int *h_depth) {
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

    /* TODO: Choose appropriate block/grid sizes */
    int blockSize = 256;
    int gridSize  = (V + blockSize - 1) / blockSize;

    while (h_found_new) {
        h_found_new = 0;
        CUDA_CHECK(cudaMemcpy(d_found_new, &h_found_new, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_new_frontier, 0, V * sizeof(int)));

        /* TODO: Launch SpMV BFS kernel */
        spmv_bfs_kernel<<<gridSize, blockSize>>>(
            d_row_ptr, d_col_idx,
            d_frontier, d_visited, d_new_frontier,
            d_depth, level, V, d_found_new);

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
 * Main
 * ============================================================ */

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <graph.edgelist> <source_vertex>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *graph_file = argv[1];
    int source = atoi(argv[2]);

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

    /* Run BFS */
    int *depth = (int *)malloc(g.num_vertices * sizeof(int));

    /* TODO: Add timing with CUDA events */
    bfs_linalg(&g, source, depth);

    /* Print results (small graphs only) or summary */
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

    /* Write depths to stdout-compatible format for validation */
    /* Usage: ./bfs_linalg graph.edgelist 0 > output.txt
     *        python3 validate_bfs.py graph.edgelist 0 output.txt */

    free(depth);
    free_graph(&g);
    return EXIT_SUCCESS;
}
