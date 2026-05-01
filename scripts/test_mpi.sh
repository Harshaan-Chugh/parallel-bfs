#!/bin/bash
# test_mpi.sh — Full MPI BFS test suite
# Run from project root: bash scripts/test_mpi.sh
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BFS_1D="$ROOT/build/bfs_mpi_1d"
BFS_2D="$ROOT/build/bfs_mpi_2d"
VALIDATE="$ROOT/test-graphs/validate_bfs.py"
RAW="$ROOT/results/mpi_raw"
mkdir -p "$RAW"

SRUN="srun -C cpu -q interactive -t 00:05:00"

# ============================================================
# Phase 1: Correctness on tiny edgelists
# ============================================================
echo "========================================"
echo "Phase 1: Correctness (tiny edgelists)"
echo "========================================"

for g in "$ROOT"/test-graphs/tiny_*.edgelist; do
    gname=$(basename "$g")
    for np in 1 2 4; do
        # 1D
        $SRUN -n $np $BFS_1D "$g" 0 --dump 2>/dev/null | \
            awk '/^--- DEPTHS ---/{flag=1; next} flag && /^-?[0-9]+$/{print}' > /tmp/mpi1d_depths.txt
        result=$(python3 "$VALIDATE" "$g" 0 /tmp/mpi1d_depths.txt 2>&1)
        echo "1D  np=$np  $gname: $result"
    done
    for np in 1 4; do
        # 2D (perfect squares only)
        $SRUN -n $np $BFS_2D "$g" 0 --dump 2>/dev/null | \
            awk '/^--- DEPTHS ---/{flag=1; next} flag && /^-?[0-9]+$/{print}' > /tmp/mpi2d_depths.txt
        result=$(python3 "$VALIDATE" "$g" 0 /tmp/mpi2d_depths.txt 2>&1)
        echo "2D  np=$np  $gname: $result"
    done
done 2>&1 | tee "$RAW/correctness_tiny.txt"

# ============================================================
# Phase 2: Timing on medium edgelists
# ============================================================
echo ""
echo "========================================"
echo "Phase 2: Timing (medium edgelists)"
echo "========================================"

for g in "$ROOT"/test-graphs/medium_*.edgelist; do
    gname=$(basename "$g")
    echo "--- $gname ---"
    for np in 1 2 4 8 16; do
        time_1d=$($SRUN -n $np $BFS_1D "$g" 0 2>/dev/null | grep "BFS time" | awk '{print $3}')
        echo "1D  np=$np  time=$time_1d s"
    done
    for np in 1 4 9 16; do
        time_2d=$($SRUN -n $np $BFS_2D "$g" 0 2>/dev/null | grep "BFS time" | awk '{print $3}')
        echo "2D  np=$np  time=$time_2d s"
    done
done 2>&1 | tee "$RAW/timing_medium.txt"

# ============================================================
# Phase 3: Timing on large edgelists
# ============================================================
echo ""
echo "========================================"
echo "Phase 3: Timing (large edgelists)"
echo "========================================"

for g in "$ROOT"/test-graphs/large_*.edgelist; do
    gname=$(basename "$g")
    echo "--- $gname ---"
    for np in 1 2 4 8 16; do
        time_1d=$($SRUN -n $np $BFS_1D "$g" 0 2>/dev/null | grep "BFS time" | awk '{print $3}')
        echo "1D  np=$np  time=$time_1d s"
    done
    for np in 1 4 9 16; do
        time_2d=$($SRUN -n $np $BFS_2D "$g" 0 2>/dev/null | grep "BFS time" | awk '{print $3}')
        echo "2D  np=$np  time=$time_2d s"
    done
done 2>&1 | tee "$RAW/timing_large.txt"

echo ""
echo "=== edgelist testing complete ==="
