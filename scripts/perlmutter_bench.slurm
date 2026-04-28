#!/bin/bash
# Submit from the repo root with:
#   sbatch -A <your_nersc_account> scripts/perlmutter_bench.slurm

#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:45:00
#SBATCH -J parallel-bfs

set -euo pipefail

cd "$(dirname "$0")/.."

module load PrgEnv-nvidia >/dev/null 2>&1 || true
module load cudatoolkit >/dev/null 2>&1 || true

export CUDA_VISIBLE_DEVICES=0

make clean
make

python3 scripts/benchmark.py --repeats "${REPEATS:-10}" --groups medium large
