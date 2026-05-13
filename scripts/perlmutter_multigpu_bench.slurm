#!/bin/bash
# Submit from the repo root with:
#   sbatch -A <your_nersc_account> scripts/perlmutter_multigpu_bench.slurm

#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -G 4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:30:00
#SBATCH -J bfs-multigpu
#SBATCH -o results/multigpu/multigpu_%j.out
#SBATCH -e results/multigpu/multigpu_%j.err

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/multigpu

module load PrgEnv-nvidia >/dev/null 2>&1 || true
module load cudatoolkit >/dev/null 2>&1 || true

# Slurm will expose the allocated GPUs. Default to all four logical IDs when
# running outside Slurm for quick interactive testing.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

make linalg-multigpu

python3 scripts/benchmark_multigpu.py \
    --repeats "${REPEATS:-3}" \
    --gpu-counts 1 2 4 \
    --groups large \
    --snap roadNet-PA roadNet-TX roadNet-CA
