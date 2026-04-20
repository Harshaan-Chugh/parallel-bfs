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
 * Usage: ./build/bfs_graphfirst <graph.edgelist> <source_vertex>
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
 * Top-down BFS kernel (push-based).
 *
 * Each thread handles one frontier vertex and explores its neighbors.
 * Good when frontier is small.
 *
 * TODO: Implement. Each frontier vertex pushes to unvisited neighbors,
 *       using atomicCAS on visited[] to avoid duplicates.
 */
__global__ void bfs_top_down_kernel(
    const int *row_ptr,
    const int *col_idx,
    const int *frontier,      /* current frontier (queue or bitmap) */
    int        frontier_size,
    int       *visited,
    int       *new_frontier,
    int       *new_frontier_size,
    int       *depth,
    int        level,
    int        num_vertices)
{
    /* TODO: Implement top-down (push) BFS expansion */
}

/*
 * Bottom-up BFS kernel (pull-based).
 *
 * Each thread handles one unvisited vertex and checks if any neighbor
 * is in the frontier. Good when frontier is large.
 *
 * TODO: Implement. Each unvisited vertex checks neighbors for frontier
 *       membership. Can stop early after finding one parent.
 */
__global__ void bfs_bottom_up_kernel(
    const int *row_ptr,
    const int *col_idx,
    const int *frontier_bitmap, /* bitmap representation of frontier */
    int       *visited,
    int       *new_frontier_bitmap,
    int       *depth,
    int        level,
    int        num_vertices,
    int       *found_new)
{
    /* TODO: Implement bottom-up (pull) BFS expansion */
}

/* ============================================================
 * Host BFS driver
 * ============================================================ */

void bfs_graphfirst(const Graph *g, int source, int *h_depth) {
    int V = g->num_vertices;
    int E = g->num_edges;

    /* Initialize host depth array */
    for (int i = 0; i < V; i++) h_depth[i] = -1;
    h_depth[source] = 0;

    /* TODO: Allocate device memory for:
     *   - CSR graph (row_ptr, col_idx)
     *   - Frontier queue/bitmap
     *   - Visited array
     *   - Depth array
     *   - Direction-switching bookkeeping
     */

    /* TODO: Copy graph to device */

    /* TODO: BFS iteration loop with direction switching:
     *
     *   while (frontier is not empty):
     *     if (should_switch_to_bottom_up(frontier_size, edges_to_check, V)):
     *       launch bfs_bottom_up_kernel
     *     else:
     *       launch bfs_top_down_kernel
     *     swap frontiers
     *     level++
     *
     * Beamer's heuristic: switch to bottom-up when
     *   edges_to_check_from_frontier > (num_edges / alpha)
     * Switch back to top-down when
     *   frontier_size < (num_vertices / beta)
     * Typical values: alpha = 14, beta = 24
     */

    /* TODO: Copy results back to host */

    /* TODO: Free device memory */

    /* PLACEHOLDER: For now, just run BFS on CPU so the binary is testable */
    int *queue = (int *)malloc(V * sizeof(int));
    int head = 0, tail = 0;
    int *visited = (int *)calloc(V, sizeof(int));
    queue[tail++] = source;
    visited[source] = 1;

    while (head < tail) {
        int u = queue[head++];
        for (int e = g->row_ptr[u]; e < g->row_ptr[u + 1]; e++) {
            int v = g->col_idx[e];
            if (!visited[v]) {
                visited[v] = 1;
                h_depth[v] = h_depth[u] + 1;
                queue[tail++] = v;
            }
        }
    }

    free(queue);
    free(visited);
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
    bfs_graphfirst(&g, source, depth);

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

    free(depth);
    free_graph(&g);
    return EXIT_SUCCESS;
}
