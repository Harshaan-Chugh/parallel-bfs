/*
 * bfs_linalg_multigpu.cu - Multi-GPU linear-algebraic BFS prototype.
 *
 * Approach:
 *   Replicate the CSR graph on each GPU, partition output rows across GPUs,
 *   and run a bitmap pull-style SpMV BFS. After each BFS level, the host ORs
 *   the per-GPU next-frontier bitmaps and broadcasts the merged frontier to
 *   every GPU for the next level.
 *
 * This is intentionally separate from bfs_linalg.cu so the original
 * single-GPU experiments remain reproducible. It is useful for optional
 * 1/2/4-GPU scaling experiments, but it still pays host-mediated frontier
 * exchange every level and is not a production multi-GPU BFS.
 *
 * Build: make linalg-multigpu
 * Usage: ./build/bfs_linalg_multigpu <graph> <source> <num_gpus> [--dump]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#include "../test-graphs/graph_loader.h"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

typedef struct {
    int device;
    int row_start;
    int row_end;
    int *d_row_ptr;
    int *d_col_idx;
    unsigned *d_frontier_bm;
    unsigned *d_new_frontier_bm;
    int *d_visited;
    int *d_depth;
} GpuShard;

static double wall_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

__global__ void mgpu_pull_bitmap_kernel(
    const int *row_ptr,
    const int *col_idx,
    const unsigned *frontier_bm,
    unsigned *new_frontier_bm,
    int *visited,
    int *depth,
    int level,
    int row_start,
    int row_end)
{
    int i = row_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_end) return;
    if (visited[i]) return;

    int begin = row_ptr[i];
    int end = row_ptr[i + 1];
    for (int e = begin; e < end; e++) {
        int j = col_idx[e];
        unsigned word = frontier_bm[j >> 5];
        unsigned bit = 1u << (j & 31);
        if (word & bit) {
            visited[i] = 1;
            depth[i] = level;
            atomicOr(&new_frontier_bm[i >> 5], 1u << (i & 31));
            break;
        }
    }
}

static int bitmap_is_empty(const unsigned *bm, int words) {
    for (int i = 0; i < words; i++) {
        if (bm[i]) return 0;
    }
    return 1;
}

static void bfs_linalg_multigpu(const Graph *g, int source, int requested_gpus,
                                int *h_depth, int *out_levels, float *out_ms) {
    int V = g->num_vertices;
    int E = g->num_edges;
    int bm_words = (V + 31) / 32;
    int visible_gpus = 0;

    CUDA_CHECK(cudaGetDeviceCount(&visible_gpus));
    if (visible_gpus <= 0) {
        fprintf(stderr, "Error: no CUDA devices visible\n");
        exit(EXIT_FAILURE);
    }
    if (requested_gpus < 1 || requested_gpus > visible_gpus) {
        fprintf(stderr, "Error: requested %d GPUs but %d are visible\n",
                requested_gpus, visible_gpus);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < V; i++) h_depth[i] = -1;
    h_depth[source] = 0;

    unsigned *h_frontier = (unsigned *)calloc(bm_words, sizeof(unsigned));
    unsigned *h_next = (unsigned *)calloc(bm_words, sizeof(unsigned));
    unsigned *h_tmp = (unsigned *)calloc(bm_words, sizeof(unsigned));
    int *h_init_visited = (int *)calloc(V, sizeof(int));
    int *h_init_depth = (int *)malloc(V * sizeof(int));
    GpuShard *shards = (GpuShard *)calloc(requested_gpus, sizeof(GpuShard));

    if (!h_frontier || !h_next || !h_tmp || !h_init_visited || !h_init_depth || !shards) {
        fprintf(stderr, "Error: host allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < V; i++) h_init_depth[i] = -1;
    h_init_depth[source] = 0;
    h_init_visited[source] = 1;
    h_frontier[source >> 5] |= 1u << (source & 31);

    for (int gpu = 0; gpu < requested_gpus; gpu++) {
        GpuShard *s = &shards[gpu];
        s->device = gpu;
        s->row_start = (int)(((long long)V * gpu) / requested_gpus);
        s->row_end = (int)(((long long)V * (gpu + 1)) / requested_gpus);

        CUDA_CHECK(cudaSetDevice(s->device));
        CUDA_CHECK(cudaMalloc(&s->d_row_ptr, (V + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&s->d_col_idx, E * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&s->d_frontier_bm, bm_words * sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&s->d_new_frontier_bm, bm_words * sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc(&s->d_visited, V * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&s->d_depth, V * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(s->d_row_ptr, g->row_ptr, (V + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s->d_col_idx, g->col_idx, E * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s->d_frontier_bm, h_frontier,
                              bm_words * sizeof(unsigned), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s->d_visited, h_init_visited, V * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(s->d_depth, h_init_depth, V * sizeof(int),
                              cudaMemcpyHostToDevice));
    }

    for (int gpu = 0; gpu < requested_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double start_ms = wall_time_ms();
    int level = 1;
    int block_size = 256;

    while (!bitmap_is_empty(h_frontier, bm_words)) {
        memset(h_next, 0, bm_words * sizeof(unsigned));

        for (int gpu = 0; gpu < requested_gpus; gpu++) {
            GpuShard *s = &shards[gpu];
            int local_rows = s->row_end - s->row_start;
            int grid_size = (local_rows + block_size - 1) / block_size;

            CUDA_CHECK(cudaSetDevice(s->device));
            CUDA_CHECK(cudaMemcpy(s->d_frontier_bm, h_frontier,
                                  bm_words * sizeof(unsigned), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(s->d_new_frontier_bm, 0,
                                  bm_words * sizeof(unsigned)));
            if (local_rows > 0) {
                mgpu_pull_bitmap_kernel<<<grid_size, block_size>>>(
                    s->d_row_ptr, s->d_col_idx, s->d_frontier_bm,
                    s->d_new_frontier_bm, s->d_visited, s->d_depth,
                    level, s->row_start, s->row_end);
            }
        }

        for (int gpu = 0; gpu < requested_gpus; gpu++) {
            CUDA_CHECK(cudaSetDevice(shards[gpu].device));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        for (int gpu = 0; gpu < requested_gpus; gpu++) {
            GpuShard *s = &shards[gpu];
            CUDA_CHECK(cudaSetDevice(s->device));
            CUDA_CHECK(cudaMemcpy(h_tmp, s->d_new_frontier_bm,
                                  bm_words * sizeof(unsigned), cudaMemcpyDeviceToHost));
            for (int w = 0; w < bm_words; w++) {
                h_next[w] |= h_tmp[w];
            }
        }

        unsigned *swap = h_frontier;
        h_frontier = h_next;
        h_next = swap;
        level++;
    }

    for (int gpu = 0; gpu < requested_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    double stop_ms = wall_time_ms();

    for (int gpu = 0; gpu < requested_gpus; gpu++) {
        GpuShard *s = &shards[gpu];
        CUDA_CHECK(cudaSetDevice(s->device));
        CUDA_CHECK(cudaMemcpy(h_depth + s->row_start, s->d_depth + s->row_start,
                              (s->row_end - s->row_start) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    *out_levels = level - 2;
    *out_ms = (float)(stop_ms - start_ms);

    for (int gpu = 0; gpu < requested_gpus; gpu++) {
        GpuShard *s = &shards[gpu];
        CUDA_CHECK(cudaSetDevice(s->device));
        CUDA_CHECK(cudaFree(s->d_row_ptr));
        CUDA_CHECK(cudaFree(s->d_col_idx));
        CUDA_CHECK(cudaFree(s->d_frontier_bm));
        CUDA_CHECK(cudaFree(s->d_new_frontier_bm));
        CUDA_CHECK(cudaFree(s->d_visited));
        CUDA_CHECK(cudaFree(s->d_depth));
    }

    free(shards);
    free(h_frontier);
    free(h_next);
    free(h_tmp);
    free(h_init_visited);
    free(h_init_depth);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr,
                "Usage: %s <graph.edgelist|snap.txt> <source_vertex> <num_gpus> [--dump]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *graph_file = argv[1];
    int source = atoi(argv[2]);
    int num_gpus = atoi(argv[3]);
    int dump = 0;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--dump") == 0) dump = 1;
    }

    Graph g;
    if (load_graph(graph_file, &g) != 0) {
        return EXIT_FAILURE;
    }
    print_graph_info(&g);
    printf("Algorithm: multi-GPU linear-algebraic bitmap pull BFS\n");
    printf("GPUs requested: %d\n", num_gpus);

    if (source < 0 || source >= g.num_vertices) {
        fprintf(stderr, "Error: source vertex %d out of range [0, %d)\n",
                source, g.num_vertices);
        free_graph(&g);
        return EXIT_FAILURE;
    }

    int *depth = (int *)malloc(g.num_vertices * sizeof(int));
    if (!depth) {
        fprintf(stderr, "Error: depth allocation failed\n");
        free_graph(&g);
        return EXIT_FAILURE;
    }

    int levels = 0;
    float elapsed_ms = 0.0f;
    bfs_linalg_multigpu(&g, source, num_gpus, depth, &levels, &elapsed_ms);

    printf("BFS time: %.3f ms\n", elapsed_ms);
    printf("Multi-GPU levels: %d\n", levels);

    if (g.num_vertices <= 50) {
        printf("BFS depths from source %d:\n", source);
        for (int i = 0; i < g.num_vertices; i++) {
            printf("  vertex %d: depth %d\n", i, depth[i]);
        }
    } else {
        int reachable = 0;
        int max_depth = 0;
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
