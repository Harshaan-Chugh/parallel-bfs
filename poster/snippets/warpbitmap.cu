// 32 lanes scan one row of A; bitmap frontier
for (int e = row_start + lane; e < row_end; e += 32) {
    int j = col_idx[e];
    if (frontier_bm[j>>5] & (1u << (j & 31))) {
        atomicOr(&new_bm[i>>5], 1u << (i & 31));
        depth[i] = level; break;
    }
}
