#!/bin/bash
# correctness_test.sh — Validate MPI BFS against Python reference
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BFS_1D="$ROOT/build/bfs_mpi_1d"
BFS_2D="$ROOT/build/bfs_mpi_2d"
VALIDATE="$ROOT/test-graphs/validate_bfs.py"
OUTFILE="$ROOT/results/mpi_raw/correctness_tiny.txt"
mkdir -p "$ROOT/results/mpi_raw"

echo "=== 1D BFS Correctness ===" | tee "$OUTFILE"
for g in "$ROOT"/test-graphs/tiny_*.edgelist; do
    gname=$(basename "$g")
    for np in 1 2 4; do
        srun -n "$np" -C cpu -q interactive -t 00:02:00 "$BFS_1D" "$g" 0 --dump 2>/dev/null | \
            awk '/^--- DEPTHS ---/{flag=1; next} flag && /^-?[0-9]+$/{print}' > /tmp/mpi1d.txt
        res=$(python3 "$VALIDATE" "$g" 0 /tmp/mpi1d.txt 2>&1)
        echo "1D np=$np $gname: $res" | tee -a "$OUTFILE"
    done
done

echo "" | tee -a "$OUTFILE"
echo "=== 2D BFS Correctness ===" | tee -a "$OUTFILE"
for g in "$ROOT"/test-graphs/tiny_*.edgelist; do
    gname=$(basename "$g")
    for np in 1 4; do
        srun -n "$np" -C cpu -q interactive -t 00:02:00 "$BFS_2D" "$g" 0 --dump 2>/dev/null | \
            awk '/^--- DEPTHS ---/{flag=1; next} flag && /^-?[0-9]+$/{print}' > /tmp/mpi2d.txt
        res=$(python3 "$VALIDATE" "$g" 0 /tmp/mpi2d.txt 2>&1)
        echo "2D np=$np $gname: $res" | tee -a "$OUTFILE"
    done
done

echo "" | tee -a "$OUTFILE"
echo "=== Done ===" | tee -a "$OUTFILE"
