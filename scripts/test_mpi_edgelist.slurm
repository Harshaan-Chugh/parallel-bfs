#!/bin/bash
#SBATCH -J mpi_bfs_test
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -t 00:30:00

# ============================================================
# MPI BFS: Full edgelist test suite
# Correctness validation + timing on tiny/medium/large graphs
# ============================================================

set -e

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
BFS_1D="$ROOT/build/bfs_mpi_1d"
BFS_2D="$ROOT/build/bfs_mpi_2d"
VALIDATE="$ROOT/test-graphs/validate_bfs.py"
RAW="$ROOT/results/mpi_raw"
mkdir -p "$RAW"

echo "========================================"
echo "Phase 1: Correctness (tiny edgelists)"
echo "Date: $(date)"
echo "========================================"

CORRECT_FILE="$RAW/correctness.txt"
> "$CORRECT_FILE"

for g in "$ROOT"/test-graphs/tiny_*.edgelist; do
    gname=$(basename "$g")
    for np in 1 2 4; do
        srun -n "$np" "$BFS_1D" "$g" 0 --dump 2>/dev/null | \
            awk '/^--- DEPTHS ---/{flag=1; next} flag && /^-?[0-9]+$/{print}' > /tmp/mpi1d.txt
        res=$(python3 "$VALIDATE" "$g" 0 /tmp/mpi1d.txt 2>&1)
        echo "1D np=$np $gname: $res" | tee -a "$CORRECT_FILE"
    done
    for np in 1 4; do
        srun -n "$np" "$BFS_2D" "$g" 0 --dump 2>/dev/null | \
            awk '/^--- DEPTHS ---/{flag=1; next} flag && /^-?[0-9]+$/{print}' > /tmp/mpi2d.txt
        res=$(python3 "$VALIDATE" "$g" 0 /tmp/mpi2d.txt 2>&1)
        echo "2D np=$np $gname: $res" | tee -a "$CORRECT_FILE"
    done
done

echo ""
echo "========================================"
echo "Phase 2: Timing (medium edgelists)"
echo "========================================"

TIMING_FILE="$RAW/timing_medium.csv"
echo "graph,strategy,ranks,time_s,reachable,max_depth" > "$TIMING_FILE"

for g in "$ROOT"/test-graphs/medium_*.edgelist; do
    gname=$(basename "$g" .edgelist)
    echo "--- $gname ---"
    for np in 1 2 4 8 16; do
        output=$(srun -n "$np" "$BFS_1D" "$g" 0 2>/dev/null)
        time_s=$(echo "$output" | grep "BFS time" | awk '{print $3}')
        reach=$(echo "$output" | grep "Reachable" | awk '{print $2}')
        mdepth=$(echo "$output" | grep "Max depth" | awk '{print $3}')
        echo "$gname,1D,$np,$time_s,$reach,$mdepth" | tee -a "$TIMING_FILE"
    done
    for np in 1 4 9 16; do
        output=$(srun -n "$np" "$BFS_2D" "$g" 0 2>/dev/null)
        time_s=$(echo "$output" | grep "BFS time" | awk '{print $3}')
        reach=$(echo "$output" | grep "Reachable" | awk '{print $2}')
        mdepth=$(echo "$output" | grep "Max depth" | awk '{print $3}')
        echo "$gname,2D,$np,$time_s,$reach,$mdepth" | tee -a "$TIMING_FILE"
    done
done

echo ""
echo "========================================"
echo "Phase 3: Timing (large edgelists)"
echo "========================================"

TIMING_FILE_L="$RAW/timing_large.csv"
echo "graph,strategy,ranks,time_s,reachable,max_depth" > "$TIMING_FILE_L"

for g in "$ROOT"/test-graphs/large_*.edgelist; do
    gname=$(basename "$g" .edgelist)
    echo "--- $gname ---"
    for np in 1 2 4 8 16; do
        output=$(srun -n "$np" "$BFS_1D" "$g" 0 2>/dev/null)
        time_s=$(echo "$output" | grep "BFS time" | awk '{print $3}')
        reach=$(echo "$output" | grep "Reachable" | awk '{print $2}')
        mdepth=$(echo "$output" | grep "Max depth" | awk '{print $3}')
        echo "$gname,1D,$np,$time_s,$reach,$mdepth" | tee -a "$TIMING_FILE_L"
    done
    for np in 1 4 9 16; do
        output=$(srun -n "$np" "$BFS_2D" "$g" 0 2>/dev/null)
        time_s=$(echo "$output" | grep "BFS time" | awk '{print $3}')
        reach=$(echo "$output" | grep "Reachable" | awk '{print $2}')
        mdepth=$(echo "$output" | grep "Max depth" | awk '{print $3}')
        echo "$gname,2D,$np,$time_s,$reach,$mdepth" | tee -a "$TIMING_FILE_L"
    done
done

echo ""
echo "=== All edgelist tests complete ==="
echo "Date: $(date)"
