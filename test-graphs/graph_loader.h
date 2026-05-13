#ifndef GRAPH_LOADER_H
#define GRAPH_LOADER_H

/*
 * graph_loader.h - Shared graph loading utility for BFS implementations
 *
 * Reads .edgelist format:
 *   # comments
 *   <num_vertices> <num_edges>    (undirected edge count)
 *   <src> <dst>
 *   ...
 *
 * Also supports SNAP-style .txt/.tab files:
 *   # comments
 *   <src> <dst>
 *   ...
 * Vertex count is inferred from the maximum ID. Each input edge is stored in
 * both directions, matching the undirected treatment used by the project
 * .edgelist files.
 *
 * Builds CSR (Compressed Sparse Row) representation, which is needed by:
 *   - Graph-first BFS: adjacency traversal
 *   - Linear algebraic BFS: sparse matrix representation for SpMV
 *
 * Usage:
 *   Graph g;
 *   load_graph("path/to/file.edgelist", &g);
 *   // ... use g.row_ptr, g.col_idx, g.num_vertices, g.num_edges ...
 *   free_graph(&g);
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int num_vertices;
    int num_edges;      /* Total directed edges (2x the undirected count) */
    int *row_ptr;       /* CSR row pointers: size num_vertices + 1 */
    int *col_idx;       /* CSR column indices: size num_edges */
} Graph;

static int graph_loader_is_snap_format(const char *filename) {
    const char *ext = strrchr(filename, '.');
    if (!ext) return 0;
    return (strcmp(ext, ".txt") == 0 || strcmp(ext, ".tab") == 0);
}

/*
 * Load a graph from an .edgelist file into CSR format.
 * Returns 0 on success, -1 on failure.
 */
static int load_graph(const char *filename, Graph *g) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", filename);
        return -1;
    }

    char line[256];
    int snap = graph_loader_is_snap_format(filename);
    int num_vertices = 0, num_undirected_edges = 0;
    int edge_capacity = 0;
    int edge_count = 0;
    int *src = NULL;
    int *dst = NULL;

    if (snap) {
        int max_vertex = -1;
        int u, v;

        /* First pass: infer vertex count and directed edge count. */
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            if (sscanf(line, "%d %d", &u, &v) == 2 && u != v) {
                edge_count += 2;
                if (u > max_vertex) max_vertex = u;
                if (v > max_vertex) max_vertex = v;
            }
        }

        num_vertices = max_vertex + 1;
        if (num_vertices <= 0) {
            fprintf(stderr, "Error: no edges found in %s\n", filename);
            fclose(fp);
            return -1;
        }

        edge_capacity = edge_count;
        src = (int *)malloc(sizeof(int) * edge_capacity);
        dst = (int *)malloc(sizeof(int) * edge_capacity);
        if (!src || !dst) {
            fprintf(stderr, "Error: malloc failed\n");
            fclose(fp);
            free(src); free(dst);
            return -1;
        }

        rewind(fp);
        edge_count = 0;
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            if (sscanf(line, "%d %d", &u, &v) == 2 && u != v) {
                src[edge_count] = u;
                dst[edge_count] = v;
                edge_count++;
                src[edge_count] = v;
                dst[edge_count] = u;
                edge_count++;
            }
        }
        fclose(fp);
    } else {
        int header_read = 0;

        /* First pass: read header */
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            if (sscanf(line, "%d %d", &num_vertices, &num_undirected_edges) != 2) {
                fprintf(stderr, "Error: invalid header in %s\n", filename);
                fclose(fp);
                return -1;
            }
            header_read = 1;
            break;
        }

        if (!header_read) {
            fprintf(stderr, "Error: missing header in %s\n", filename);
            fclose(fp);
            return -1;
        }

        edge_capacity = 2 * num_undirected_edges; /* Each undirected edge = 2 directed */

        /* Allocate temporary edge storage */
        src = (int *)malloc(sizeof(int) * edge_capacity);
        dst = (int *)malloc(sizeof(int) * edge_capacity);
        if (!src || !dst) {
            fprintf(stderr, "Error: malloc failed\n");
            fclose(fp);
            free(src); free(dst);
            return -1;
        }

        /* Read edges (store both directions) */
        edge_count = 0;
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            int u, v;
            if (sscanf(line, "%d %d", &u, &v) == 2) {
                if (edge_count + 2 > edge_capacity) {
                    edge_capacity = edge_capacity ? edge_capacity * 2 : 1024;
                    src = (int *)realloc(src, sizeof(int) * edge_capacity);
                    dst = (int *)realloc(dst, sizeof(int) * edge_capacity);
                    if (!src || !dst) {
                        fprintf(stderr, "Error: realloc failed\n");
                        fclose(fp);
                        free(src); free(dst);
                        return -1;
                    }
                }
                src[edge_count] = u;
                dst[edge_count] = v;
                edge_count++;
                src[edge_count] = v;
                dst[edge_count] = u;
                edge_count++;
            }
        }
        fclose(fp);
    }

    g->num_vertices = num_vertices;
    g->num_edges = edge_count; /* Actual directed edge count */

    /* Build CSR: count degrees */
    g->row_ptr = (int *)calloc(num_vertices + 1, sizeof(int));
    if (!g->row_ptr) {
        fprintf(stderr, "Error: malloc failed\n");
        free(src); free(dst);
        return -1;
    }

    for (int i = 0; i < edge_count; i++) {
        g->row_ptr[src[i] + 1]++;
    }

    /* Prefix sum */
    for (int i = 1; i <= num_vertices; i++) {
        g->row_ptr[i] += g->row_ptr[i - 1];
    }

    /* Fill col_idx */
    g->col_idx = (int *)malloc(sizeof(int) * edge_count);
    int *offset = (int *)calloc(num_vertices, sizeof(int));
    if (!g->col_idx || !offset) {
        fprintf(stderr, "Error: malloc failed\n");
        free(src); free(dst); free(g->row_ptr);
        return -1;
    }

    for (int i = 0; i < edge_count; i++) {
        int row = src[i];
        int pos = g->row_ptr[row] + offset[row];
        g->col_idx[pos] = dst[i];
        offset[row]++;
    }

    free(src);
    free(dst);
    free(offset);

    return 0;
}

static void free_graph(Graph *g) {
    if (g->row_ptr) free(g->row_ptr);
    if (g->col_idx) free(g->col_idx);
    g->row_ptr = NULL;
    g->col_idx = NULL;
    g->num_vertices = 0;
    g->num_edges = 0;
}

static void print_graph_info(const Graph *g) {
    printf("Graph: %d vertices, %d directed edges (avg degree: %.1f)\n",
           g->num_vertices, g->num_edges,
           (double)g->num_edges / g->num_vertices);
}

#endif /* GRAPH_LOADER_H */
