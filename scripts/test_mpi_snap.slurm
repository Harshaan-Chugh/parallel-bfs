#!/bin/bash
#SBATCH -J mpi_snap_test
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --ntasks-per-node=16
#SBATCH -t 01:00:00
#SBATCH -D .
#SBATCH -o results/mpi_raw/snap_test_%j.out
#SBATCH -e results/mpi_raw/snap_test_%j.err

# ============================================================
# MPI BFS: SNAP road network benchmark
# Strong-scaling on roadNet-PA, roadNet-TX, roadNet-CA
# ============================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BFS_1D="$ROOT/build/bfs_mpi_1d"
BFS_2D="$ROOT/build/bfs_mpi_2d"
SNAP="$ROOT/snap-graphs"
RAW="$ROOT/results/mpi_raw"
mkdir -p "$RAW"

SOURCE=0

echo "========================================"
echo "SNAP Road Network MPI BFS Benchmarks"
echo "Date: $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "========================================"

# ---- Phase 1: Correctness (1 rank vs Python reference) ----
echo ""
echo "--- Correctness: 1D np=1 ---"
CORRECT_FILE="$RAW/snap_correctness.txt"
> "$CORRECT_FILE"

for dataset in roadNet-PA roadNet-TX roadNet-CA; do
    graph="$SNAP/${dataset}.txt"
    if [ ! -f "$graph" ]; then
        echo "SKIP: $graph not found" | tee -a "$CORRECT_FILE"
        continue
    fi
    srun -n 1 "$BFS_1D" "$graph" "$SOURCE" --dump 2>/dev/null | \
        awk '/^--- DEPTHS ---/{flag=1; next} flag && /^-?[0-9]+$/{print}' > /tmp/snap_depths.txt
    # Just check reachable count and depth consistency (Python ref too slow for 2M vertices)
    total=$(wc -l < /tmp/snap_depths.txt)
    reachable=$(grep -c -v "^-1$" /tmp/snap_depths.txt || true)
    max_d=$(sort -n /tmp/snap_depths.txt | tail -1)
    echo "$dataset: $reachable/$total reachable, max_depth=$max_d" | tee -a "$CORRECT_FILE"

    # Rank-count invariance: compare 1-rank output to 4-rank output
    srun -n 4 "$BFS_1D" "$graph" "$SOURCE" --dump 2>/dev/null | \
        awk '/^--- DEPTHS ---/{flag=1; next} flag && /^-?[0-9]+$/{print}' > /tmp/snap_depths_4.txt
    if diff -q /tmp/snap_depths.txt /tmp/snap_depths_4.txt > /dev/null 2>&1; then
        echo "$dataset: 1D np=1 vs np=4 MATCH" | tee -a "$CORRECT_FILE"
    else
        echo "$dataset: 1D np=1 vs np=4 MISMATCH!" | tee -a "$CORRECT_FILE"
    fi

    # 1D vs 2D comparison
    srun -n 4 "$BFS_2D" "$graph" "$SOURCE" --dump 2>/dev/null | \
        awk '/^--- DEPTHS ---/{flag=1; next} flag && /^-?[0-9]+$/{print}' > /tmp/snap_depths_2d.txt
    if diff -q /tmp/snap_depths.txt /tmp/snap_depths_2d.txt > /dev/null 2>&1; then
        echo "$dataset: 1D vs 2D MATCH" | tee -a "$CORRECT_FILE"
    else
        echo "$dataset: 1D vs 2D MISMATCH!" | tee -a "$CORRECT_FILE"
    fi
done

# ---- Phase 2: Strong-scaling timing ----
echo ""
echo "========================================"
echo "SNAP Strong-Scaling Timing"
echo "========================================"

TIMING_FILE="$RAW/timing_snap.csv"
echo "graph,strategy,ranks,time_s,reachable,max_depth" > "$TIMING_FILE"

# Run 3 trials per config, record all
for dataset in roadNet-PA roadNet-TX roadNet-CA; do
    graph="$SNAP/${dataset}.txt"
    if [ ! -f "$graph" ]; then
        echo "SKIP: $graph"
        continue
    fi
    echo ""
    echo "=== $dataset ==="

    for np in 1 2 4 8 16 32 64; do
        for trial in 1 2 3; do
            output=$(srun -n "$np" "$BFS_1D" "$graph" "$SOURCE" 2>/dev/null)
            time_s=$(echo "$output" | grep "BFS time" | awk '{print $3}')
            reach=$(echo "$output" | grep "Reachable" | awk '{print $2}')
            mdepth=$(echo "$output" | grep "Max depth" | awk '{print $3}')
            echo "$dataset,1D,$np,$time_s,$reach,$mdepth" | tee -a "$TIMING_FILE"
        done
    done

    for np in 1 4 9 16 25 36 49 64; do
        for trial in 1 2 3; do
            output=$(srun -n "$np" "$BFS_2D" "$graph" "$SOURCE" 2>/dev/null)
            time_s=$(echo "$output" | grep "BFS time" | awk '{print $3}')
            reach=$(echo "$output" | grep "Reachable" | awk '{print $2}')
            mdepth=$(echo "$output" | grep "Max depth" | awk '{print $3}')
            echo "$dataset,2D,$np,$time_s,$reach,$mdepth" | tee -a "$TIMING_FILE"
        done
    done
done

echo ""
echo "=== SNAP testing complete ==="
echo "Date: $(date)"
