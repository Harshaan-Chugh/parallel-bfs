#ifndef GRAPH_DIST_H
#define GRAPH_DIST_H

/*
 * graph_dist.h — Distributed graph loader for MPI BFS
 *
 * Supports two input formats:
 *   1. .edgelist format:  # comments, then "V E" header, then "src dst" lines
 *   2. SNAP .txt format:  # comments, then "src\tdst" lines (tab-separated,
 *                         no explicit header; V and E are inferred)
 *
 * The format is auto-detected: if the file extension is .txt or .tab, SNAP
 * format is assumed; otherwise .edgelist format is assumed.
 *
 * Builds local CSR partitions for:
 *   - 1D vertex partitioning (contiguous row blocks)
 *   - 2D edge partitioning  (submatrix blocks on a process grid)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/* ------------------------------------------------------------------ */
/*  Data Structures                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    int    global_V;        /* total number of vertices                  */
    long   global_E;        /* total directed edges (2× undirected)      */
    int    local_V;         /* vertices owned by this rank               */
    int    v_start, v_end;  /* [v_start, v_end) global vertex range      */
    int   *row_ptr;         /* local CSR row pointers (size local_V + 1) */
    int   *col_idx;         /* local CSR column indices (global IDs)     */
    int    local_E;         /* number of local directed edges            */
} DistGraph1D;

typedef struct {
    int    global_V;
    long   global_E;
    /* Block owned by rank (r,c) in a grid_r × grid_c process grid      */
    int    grid_r, grid_c;  /* process grid dimensions                   */
    int    my_pr,  my_pc;   /* this rank's grid coordinates              */
    int    row_start, row_end;  /* [row_start, row_end) row block        */
    int    col_start, col_end;  /* [col_start, col_end) col block        */
    int    local_rows;
    int    local_cols;
    int   *row_ptr;         /* local CSR for this submatrix block        */
    int   *col_idx;         /* column indices local to [col_start,col_end) */
    int    local_E;
} DistGraph2D;

/* ------------------------------------------------------------------ */
/*  Edge list helpers (rank 0 reads, then broadcasts)                   */
/* ------------------------------------------------------------------ */

/* Detect format from filename extension */
__attribute__((unused))
static int is_snap_format(const char *filename) {
    const char *ext = strrchr(filename, '.');
    if (!ext) return 0;
    return (strcmp(ext, ".txt") == 0 || strcmp(ext, ".tab") == 0);
}

/*
 * Read all edges on rank 0.  Returns edge arrays via pointers.
 * For SNAP format: auto-discovers V, renumbers vertices 0..V-1.
 * For edgelist format: reads header V E, uses IDs as-is.
 * All edges are stored as *directed* (both directions).
 */
__attribute__((unused))
static int read_edges_rank0(const char *filename,
                            int **out_src, int **out_dst,
                            int *out_V, long *out_E)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) { fprintf(stderr, "Error: cannot open %s\n", filename); return -1; }

    int snap = is_snap_format(filename);
    char line[512];
    int cap = 1 << 20;   /* initial edge capacity */
    int *src = (int *)malloc(sizeof(int) * cap);
    int *dst = (int *)malloc(sizeof(int) * cap);
    long count = 0;

    int V = 0;

    if (!snap) {
        /* .edgelist: first non-comment line is "V E" */
        int header_read = 0;
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            if (!header_read) {
                int ne;
                sscanf(line, "%d %d", &V, &ne);
                header_read = 1;
                break;
            }
        }
    }

    /* Read edges */
    int max_id = -1;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        int u, v;
        if (sscanf(line, "%d %d", &u, &v) != 2) continue;
        if (u == v) continue;  /* skip self-loops */

        /* Ensure capacity (we add 2 edges per line) */
        if (count + 2 > cap) {
            cap *= 2;
            src = (int *)realloc(src, sizeof(int) * cap);
            dst = (int *)realloc(dst, sizeof(int) * cap);
        }
        src[count] = u;  dst[count] = v;  count++;
        src[count] = v;  dst[count] = u;  count++;

        if (u > max_id) max_id = u;
        if (v > max_id) max_id = v;
    }
    fclose(fp);

    if (snap) {
        V = max_id + 1;  /* SNAP: vertex IDs are 0-based integers */
    }

    *out_src = src;
    *out_dst = dst;
    *out_V   = V;
    *out_E   = count;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  1D Vertex Partitioning                                             */
/* ------------------------------------------------------------------ */

/*
 * Load and distribute a graph using 1D vertex partitioning.
 * Rank 0 reads the file, broadcasts metadata, then each rank
 * filters edges to build its local CSR.
 */
__attribute__((unused))
static int load_graph_1d(const char *filename, DistGraph1D *g,
                         int rank, int nprocs)
{
    int    V = 0;
    long   E = 0;
    int   *all_src = NULL, *all_dst = NULL;

    if (rank == 0) {
        if (read_edges_rank0(filename, &all_src, &all_dst, &V, &E) != 0)
            return -1;
    }

    /* Broadcast global metadata */
    MPI_Bcast(&V, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&E, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    /* Broadcast edge arrays to all ranks */
    if (rank != 0) {
        all_src = (int *)malloc(sizeof(int) * E);
        all_dst = (int *)malloc(sizeof(int) * E);
    }
    MPI_Bcast(all_src, (int)E, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_dst, (int)E, MPI_INT, 0, MPI_COMM_WORLD);

    /* Compute this rank's vertex range */
    int base = V / nprocs;
    int rem  = V % nprocs;
    int v_start, v_end;
    if (rank < rem) {
        v_start = rank * (base + 1);
        v_end   = v_start + base + 1;
    } else {
        v_start = rem * (base + 1) + (rank - rem) * base;
        v_end   = v_start + base;
    }
    int local_V = v_end - v_start;

    g->global_V = V;
    g->global_E = E;
    g->local_V  = local_V;
    g->v_start  = v_start;
    g->v_end    = v_end;

    /* Count local edges (edges whose source is in [v_start, v_end)) */
    int local_E = 0;
    for (long i = 0; i < E; i++) {
        if (all_src[i] >= v_start && all_src[i] < v_end)
            local_E++;
    }
    g->local_E = local_E;

    /* Build local CSR */
    g->row_ptr = (int *)calloc(local_V + 1, sizeof(int));

    /* Count degrees */
    for (long i = 0; i < E; i++) {
        int s = all_src[i];
        if (s >= v_start && s < v_end)
            g->row_ptr[s - v_start + 1]++;
    }

    /* Prefix sum */
    for (int i = 1; i <= local_V; i++)
        g->row_ptr[i] += g->row_ptr[i - 1];

    /* Fill col_idx */
    g->col_idx = (int *)malloc(sizeof(int) * local_E);
    int *offset = (int *)calloc(local_V, sizeof(int));

    for (long i = 0; i < E; i++) {
        int s = all_src[i];
        if (s >= v_start && s < v_end) {
            int lr = s - v_start;
            int pos = g->row_ptr[lr] + offset[lr];
            g->col_idx[pos] = all_dst[i];
            offset[lr]++;
        }
    }

    free(offset);
    free(all_src);
    free(all_dst);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  2D Edge Partitioning                                               */
/* ------------------------------------------------------------------ */

/*
 * Compute integer square root; returns 0 if nprocs is not a
 * perfect square.
 */
__attribute__((unused))
static int isqrt_exact(int n) {
    for (int i = 1; i * i <= n; i++)
        if (i * i == n) return i;
    return 0;
}

/*
 * Load and distribute a graph using 2D edge partitioning.
 * Requires nprocs to be a perfect square.
 * The adjacency matrix is conceptually split into a grid_dim × grid_dim
 * block layout.  Rank (r,c) owns the block at row-block r, col-block c.
 */
__attribute__((unused))
static int load_graph_2d(const char *filename, DistGraph2D *g,
                         int rank, int nprocs)
{
    int grid_dim = isqrt_exact(nprocs);
    if (grid_dim == 0) {
        if (rank == 0)
            fprintf(stderr, "Error: 2D BFS requires a perfect-square number "
                            "of ranks (got %d)\n", nprocs);
        return -1;
    }

    int    V = 0;
    long   E = 0;
    int   *all_src = NULL, *all_dst = NULL;

    if (rank == 0) {
        if (read_edges_rank0(filename, &all_src, &all_dst, &V, &E) != 0)
            return -1;
    }

    MPI_Bcast(&V, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&E, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        all_src = (int *)malloc(sizeof(int) * E);
        all_dst = (int *)malloc(sizeof(int) * E);
    }
    MPI_Bcast(all_src, (int)E, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_dst, (int)E, MPI_INT, 0, MPI_COMM_WORLD);

    /* Grid coordinates */
    int my_pr = rank / grid_dim;
    int my_pc = rank % grid_dim;

    /* Row block for row-index my_pr */
    int rbase = V / grid_dim, rrem = V % grid_dim;
    int row_start, row_end;
    if (my_pr < rrem) {
        row_start = my_pr * (rbase + 1);
        row_end   = row_start + rbase + 1;
    } else {
        row_start = rrem * (rbase + 1) + (my_pr - rrem) * rbase;
        row_end   = row_start + rbase;
    }

    /* Column block for col-index my_pc */
    int cbase = V / grid_dim, crem = V % grid_dim;
    int col_start, col_end;
    if (my_pc < crem) {
        col_start = my_pc * (cbase + 1);
        col_end   = col_start + cbase + 1;
    } else {
        col_start = crem * (cbase + 1) + (my_pc - crem) * cbase;
        col_end   = col_start + cbase;
    }

    int local_rows = row_end - row_start;
    int local_cols = col_end - col_start;

    g->global_V  = V;
    g->global_E  = E;
    g->grid_r    = grid_dim;
    g->grid_c    = grid_dim;
    g->my_pr     = my_pr;
    g->my_pc     = my_pc;
    g->row_start = row_start;
    g->row_end   = row_end;
    g->col_start = col_start;
    g->col_end   = col_end;
    g->local_rows = local_rows;
    g->local_cols = local_cols;

    /* Count edges in this block: src in [row_start,row_end), dst in [col_start,col_end) */
    int local_E = 0;
    for (long i = 0; i < E; i++) {
        if (all_src[i] >= row_start && all_src[i] < row_end &&
            all_dst[i] >= col_start && all_dst[i] < col_end)
            local_E++;
    }
    g->local_E = local_E;

    /* Build local CSR (row indices are local, col indices are local to col block) */
    g->row_ptr = (int *)calloc(local_rows + 1, sizeof(int));

    for (long i = 0; i < E; i++) {
        int s = all_src[i], d = all_dst[i];
        if (s >= row_start && s < row_end &&
            d >= col_start && d < col_end)
            g->row_ptr[s - row_start + 1]++;
    }
    for (int i = 1; i <= local_rows; i++)
        g->row_ptr[i] += g->row_ptr[i - 1];

    g->col_idx = (int *)malloc(sizeof(int) * (local_E > 0 ? local_E : 1));
    int *offset = (int *)calloc(local_rows, sizeof(int));

    for (long i = 0; i < E; i++) {
        int s = all_src[i], d = all_dst[i];
        if (s >= row_start && s < row_end &&
            d >= col_start && d < col_end) {
            int lr = s - row_start;
            int pos = g->row_ptr[lr] + offset[lr];
            g->col_idx[pos] = d - col_start;  /* store as local col index */
            offset[lr]++;
        }
    }

    free(offset);
    free(all_src);
    free(all_dst);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Cleanup                                                            */
/* ------------------------------------------------------------------ */

__attribute__((unused))
static void free_dist_graph_1d(DistGraph1D *g) {
    free(g->row_ptr); g->row_ptr = NULL;
    free(g->col_idx); g->col_idx = NULL;
}

__attribute__((unused))
static void free_dist_graph_2d(DistGraph2D *g) {
    free(g->row_ptr); g->row_ptr = NULL;
    free(g->col_idx); g->col_idx = NULL;
}

#endif /* GRAPH_DIST_H */
