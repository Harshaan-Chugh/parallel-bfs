#!/bin/bash
#SBATCH -J mpi_bfs_bench
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --ntasks-per-node=16
#SBATCH -t 00:30:00
#SBATCH -o results/mpi_bench_%j.out
#SBATCH -e results/mpi_bench_%j.err

# ============================================================
# MPI BFS Strong-Scaling Benchmark on Perlmutter CPU Nodes
#
# Runs 1D and 2D BFS on all three SNAP road networks with
# increasing rank counts: 1, 4, 9, 16, 25, 36, 49, 64
# (perfect squares for 2D compatibility).
# ============================================================

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BFS_1D="$ROOT_DIR/build/bfs_mpi_1d"
BFS_2D="$ROOT_DIR/build/bfs_mpi_2d"
SNAP_DIR="$ROOT_DIR/snap-graphs"

SOURCE=0
GRAPHS=("roadNet-PA" "roadNet-TX" "roadNet-CA")
# Perfect squares so 2D works too
RANKS=(1 4 9 16 25 36 49 64)

mkdir -p "$ROOT_DIR/results"

echo "=== MPI BFS Strong-Scaling Benchmark ==="
echo "Date: $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo ""

for graph_name in "${GRAPHS[@]}"; do
    graph="$SNAP_DIR/${graph_name}.txt"
    if [ ! -f "$graph" ]; then
        echo "SKIP: $graph not found"
        continue
    fi

    echo "=========================================="
    echo "Graph: $graph_name"
    echo "=========================================="

    for np in "${RANKS[@]}"; do
        echo ""
        echo "--- Ranks: $np ---"

        echo "[1D]"
        srun -n "$np" "$BFS_1D" "$graph" "$SOURCE" 2>&1 | grep -E "(time|Reachable|Max depth)"

        echo "[2D]"
        srun -n "$np" "$BFS_2D" "$graph" "$SOURCE" 2>&1 | grep -E "(time|Reachable|Max depth)"
    done
    echo ""
done

echo "=== Benchmark complete ==="
