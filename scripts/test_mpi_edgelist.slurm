#!/bin/bash
#SBATCH -J mpi_bfs_test
#SBATCH -A m4341
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH -t 02:00:00
#SBATCH -D .
#SBATCH -o slurm-edgelist-%j.out
#SBATCH -e slurm-edgelist-%j.err

# ============================================================
# MPI BFS: Full edgelist test suite
# Correctness validation + timing on tiny/medium/large graphs
# ============================================================

set -euo pipefail
shopt -s nullglob

# -----------------------------
# Resolve repo root
# -----------------------------
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT="$SLURM_SUBMIT_DIR"
else
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

RAW="$ROOT/results/mpi_raw"
TMP="$RAW/tmp_${SLURM_JOB_ID:-manual}"
mkdir -p "$RAW" "$TMP"

# Also copy this job's stdout/stderr into results/mpi_raw.
# Slurm itself writes to slurm-edgelist-<jobid>.out/.err in the submit dir.
exec > >(tee -a "$RAW/edgelist_test_${SLURM_JOB_ID:-manual}.out")
exec 2> >(tee -a "$RAW/edgelist_test_${SLURM_JOB_ID:-manual}.err" >&2)

trap 'rm -rf "$TMP"' EXIT

BFS_1D="$ROOT/build/bfs_mpi_1d"
BFS_2D="$ROOT/build/bfs_mpi_2d"
VALIDATE="$ROOT/test-graphs/validate_bfs.py"

echo "========================================"
echo "MPI BFS edgelist test suite"
echo "Date: $(date)"
echo "ROOT: $ROOT"
echo "RAW:  $RAW"
echo "TMP:  $TMP"
echo "Job:  ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURM_JOB_NODELIST:-local}"
echo "========================================"
echo ""

cd "$ROOT"

# -----------------------------
# Build if needed
# -----------------------------
if [[ ! -x "$BFS_1D" || ! -x "$BFS_2D" ]]; then
    echo "MPI binaries missing or not executable."
    echo "Running make..."
    make
fi

[[ -x "$BFS_1D" ]] || { echo "ERROR: missing executable $BFS_1D"; exit 1; }
[[ -x "$BFS_2D" ]] || { echo "ERROR: missing executable $BFS_2D"; exit 1; }
[[ -f "$VALIDATE" ]] || { echo "ERROR: missing validator $VALIDATE"; exit 1; }

# Support either .edgelist or .ed, since some earlier commands used .ed.
tiny_graphs=("$ROOT"/test-graphs/tiny_*.edgelist "$ROOT"/test-graphs/tiny_*.ed)
medium_graphs=("$ROOT"/test-graphs/medium_*.edgelist "$ROOT"/test-graphs/medium_*.ed)
large_graphs=("$ROOT"/test-graphs/large_*.edgelist "$ROOT"/test-graphs/large_*.ed)

(( ${#tiny_graphs[@]} > 0 )) || { echo "ERROR: no tiny graphs found in test-graphs/"; exit 1; }
(( ${#medium_graphs[@]} > 0 )) || { echo "ERROR: no medium graphs found in test-graphs/"; exit 1; }
(( ${#large_graphs[@]} > 0 )) || { echo "ERROR: no large graphs found in test-graphs/"; exit 1; }

echo "Found ${#tiny_graphs[@]} tiny graphs"
echo "Found ${#medium_graphs[@]} medium graphs"
echo "Found ${#large_graphs[@]} large graphs"
echo ""

run_mpi() {
    local np="$1"
    shift
    srun --cpu-bind=cores -n "$np" "$@"
}

extract_metrics_or_die() {
    local output="$1"
    local graph="$2"
    local strategy="$3"
    local np="$4"
    local trial="$5"

    local time_s
    local reach
    local mdepth

    time_s="$(awk '/BFS time/{print $3; exit}' <<< "$output")"
    reach="$(awk '/Reachable/{print $2; exit}' <<< "$output")"
    mdepth="$(awk '/Max depth/{print $3; exit}' <<< "$output")"

    if [[ -z "$time_s" || -z "$reach" || -z "$mdepth" ]]; then
        echo "ERROR: failed to parse metrics"
        echo "graph=$graph strategy=$strategy np=$np trial=$trial"
        echo "Raw output:"
        echo "$output"
        exit 1
    fi

    echo "$time_s,$reach,$mdepth"
}

write_medians() {
    local input_csv="$1"
    local output_csv="$2"

    python3 - "$input_csv" "$output_csv" <<'PY'
import csv
import statistics
import sys
from collections import Counter

inp, outp = sys.argv[1], sys.argv[2]

rows = []
with open(inp, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            row["time_s"] = float(row["time_s"])
            int(row["ranks"])
        except Exception:
            continue
        rows.append(row)

groups = {}
for row in rows:
    key = (row["graph"], row["strategy"], int(row["ranks"]))
    groups.setdefault(key, []).append(row)

def most_common(values):
    return Counter(values).most_common(1)[0][0]

with open(outp, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "graph",
        "strategy",
        "ranks",
        "median_time_s",
        "reachable",
        "max_depth",
        "trials"
    ])

    for (graph, strategy, ranks), rs in sorted(groups.items()):
        times = [r["time_s"] for r in rs]
        reachable = most_common([r["reachable"] for r in rs])
        max_depth = most_common([r["max_depth"] for r in rs])

        writer.writerow([
            graph,
            strategy,
            ranks,
            f"{statistics.median(times):.6f}",
            reachable,
            max_depth,
            len(times)
        ])
PY
}

# ============================================================
# Phase 1: Correctness
# ============================================================

echo "========================================"
echo "Phase 1: Correctness validation"
echo "========================================"

CORRECT_FILE="$RAW/correctness.txt"
: > "$CORRECT_FILE"

for g in "${tiny_graphs[@]}"; do
    gname="$(basename "$g")"
    echo "--- Correctness: $gname ---"

    for np in 1 2 4; do
        out_file="$TMP/mpi1d_${np}.txt"

        run_mpi "$np" "$BFS_1D" "$g" 0 --dump \
            | awk '/^--- DEPTHS ---/{flag=1; next} flag && /^-?[0-9]+$/{print}' \
            > "$out_file"

        res="$(python3 "$VALIDATE" "$g" 0 "$out_file" 2>&1)"
        echo "1D,np=$np,graph=$gname,result=$res" | tee -a "$CORRECT_FILE"
    done

    for np in 1 4; do
        out_file="$TMP/mpi2d_${np}.txt"

        run_mpi "$np" "$BFS_2D" "$g" 0 --dump \
            | awk '/^--- DEPTHS ---/{flag=1; next} flag && /^-?[0-9]+$/{print}' \
            > "$out_file"

        res="$(python3 "$VALIDATE" "$g" 0 "$out_file" 2>&1)"
        echo "2D,np=$np,graph=$gname,result=$res" | tee -a "$CORRECT_FILE"
    done
done

echo ""
echo "Correctness results written to:"
echo "$CORRECT_FILE"
echo ""

# ============================================================
# Phase 2: Medium timing
# ============================================================

echo "========================================"
echo "Phase 2: Timing medium edgelists"
echo "========================================"

TRIALS=10

TIMING_FILE="$RAW/timing_medium_trials.csv"
TIMING_MEDIAN="$RAW/timing_medium_median.csv"

echo "graph,strategy,ranks,trial,time_s,reachable,max_depth" > "$TIMING_FILE"

for g in "${medium_graphs[@]}"; do
    gbase="$(basename "$g")"
    gname="${gbase%.*}"

    echo "--- Medium graph: $gname ---"

    for np in 1 2 4 8 16; do
        for trial in $(seq 1 "$TRIALS"); do
            output="$(run_mpi "$np" "$BFS_1D" "$g" 0)"
            metrics="$(extract_metrics_or_die "$output" "$gname" "1D" "$np" "$trial")"
            echo "$gname,1D,$np,$trial,$metrics" | tee -a "$TIMING_FILE"
        done
    done

    for np in 1 4 9 16; do
        for trial in $(seq 1 "$TRIALS"); do
            output="$(run_mpi "$np" "$BFS_2D" "$g" 0)"
            metrics="$(extract_metrics_or_die "$output" "$gname" "2D" "$np" "$trial")"
            echo "$gname,2D,$np,$trial,$metrics" | tee -a "$TIMING_FILE"
        done
    done
done

write_medians "$TIMING_FILE" "$TIMING_MEDIAN"

echo ""
echo "Medium trial CSV written to:"
echo "$TIMING_FILE"
echo "Medium median CSV written to:"
echo "$TIMING_MEDIAN"
echo ""

# ============================================================
# Phase 3: Large timing
# ============================================================

echo "========================================"
echo "Phase 3: Timing large edgelists"
echo "========================================"

TIMING_FILE_L="$RAW/timing_large_trials.csv"
TIMING_MEDIAN_L="$RAW/timing_large_median.csv"

echo "graph,strategy,ranks,trial,time_s,reachable,max_depth" > "$TIMING_FILE_L"

for g in "${large_graphs[@]}"; do
    gbase="$(basename "$g")"
    gname="${gbase%.*}"

    echo "--- Large graph: $gname ---"

    for np in 1 2 4 8 16; do
        for trial in $(seq 1 "$TRIALS"); do
            output="$(run_mpi "$np" "$BFS_1D" "$g" 0)"
            metrics="$(extract_metrics_or_die "$output" "$gname" "1D" "$np" "$trial")"
            echo "$gname,1D,$np,$trial,$metrics" | tee -a "$TIMING_FILE_L"
        done
    done

    for np in 1 4 9 16; do
        for trial in $(seq 1 "$TRIALS"); do
            output="$(run_mpi "$np" "$BFS_2D" "$g" 0)"
            metrics="$(extract_metrics_or_die "$output" "$gname" "2D" "$np" "$trial")"
            echo "$gname,2D,$np,$trial,$metrics" | tee -a "$TIMING_FILE_L"
        done
    done
done

write_medians "$TIMING_FILE_L" "$TIMING_MEDIAN_L"

echo ""
echo "Large trial CSV written to:"
echo "$TIMING_FILE_L"
echo "Large median CSV written to:"
echo "$TIMING_MEDIAN_L"
echo ""

echo "========================================"
echo "All edgelist tests complete"
echo "Date: $(date)"
echo "Raw results directory:"
echo "$RAW"
echo "========================================"