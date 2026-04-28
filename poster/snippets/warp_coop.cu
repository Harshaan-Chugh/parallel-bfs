// One warp = 32 threads working on ONE vertex i.
// row_start/row_end = neighbor indices in col_idx; lane = thread id 0..31.
for (int e = row_start + lane; e < row_end; e += 32) {
    if (frontier[col_idx[e]]) {   // neighbor on frontier?
        hit = 1;
        break;
    }
}
// Did ANY thread see a hit? ballot_sync collects one bit per thread.
unsigned mask = __ballot_sync(0xFFFFFFFF, hit);
// Lane 0 writes once if any lane found a frontier neighbor.
if (mask && lane == 0) {
    new_frontier[i] = 1;
    visited[i] = 1;
    depth[i] = level;
    atomicOr(found_new, 1);
}
