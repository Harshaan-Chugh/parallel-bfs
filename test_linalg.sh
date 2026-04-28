#!/bin/bash
# test_linalg.sh - Test all kernel variants on all graph sizes
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LINALG="$ROOT_DIR/build/bfs_linalg"
TEST_DIR="$ROOT_DIR/test-graphs"

export CUDA_VISIBLE_DEVICES=0

echo "============================================="
echo "SpMV BFS - Testing all kernel variants"
echo "============================================="

# ---- Tiny graphs (correctness check) ----
for variant in baseline warp bitmap pushpull warpbitmap; do
    echo ""
    echo "===== Variant: $variant ====="
    echo ""
    echo "--- Tiny graphs ---"
    for graph in "$TEST_DIR"/tiny_*.edgelist; do
        echo "  $(basename "$graph")"
        srun -G 1 "$LINALG" "$graph" 0 "$variant"
        echo ""
    done

    echo "--- Medium graphs ---"
    for graph in "$TEST_DIR"/medium_*.edgelist; do
        echo "  $(basename "$graph")"
        srun -G 1 "$LINALG" "$graph" 0 "$variant"
        echo ""
    done

    echo "--- Large graphs ---"
    for graph in "$TEST_DIR"/large_*.edgelist; do
        echo "  $(basename "$graph")"
        srun -G 1 "$LINALG" "$graph" 0 "$variant"
        echo ""
    done
done

echo ""
echo "============================================="
echo "All tests complete!"
echo "============================================="
