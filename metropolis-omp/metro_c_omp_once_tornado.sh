#!/bin/bash
#SBATCH --job-name=metro_c_omp_once
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:20:00
#SBATCH --partition=tornado
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

module purge
module load compiler/gcc/11

ROOT="$HOME/metropolis-mpi"
WORKDIR="$ROOT/metropolis-omp"
cd "$WORKDIR"

# --- сборка (verbose вывод как раньше) ---
gcc -O3 -std=c99 -D_POSIX_C_SOURCE=200112L -fopenmp -DVERBOSE \
  metropolis_omp.c -o metropolis_omp -lm

# --- параметры (можно переопределять через sbatch) ---
TOTAL_T="${1:-100000000}"
SIGMA="${2:-0.5}"
WRITE_CHAIN="${3:-1}"     # <<< ВАЖНО: 1 => пишем chain_omp_thread*.dat
OMP_THREADS="${4:-$SLURM_CPUS_PER_TASK}"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Nodes Allocated   = ${SLURM_JOB_NUM_NODES}"
echo "Tasks Allocated   = ${SLURM_NTASKS}"
echo "Cores/Task        = ${SLURM_CPUS_PER_TASK}"
echo ""
echo "Run config: OMP_NUM_THREADS=${OMP_THREADS} TOTAL_T=${TOTAL_T} SIGMA=${SIGMA} WRITE_CHAIN=${WRITE_CHAIN}"
echo ""

export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# argv: TOTAL_T sigma write_chain
srun -n 1 -c "${SLURM_CPUS_PER_TASK}" ./metropolis_omp "$TOTAL_T" "$SIGMA" "$WRITE_CHAIN"

echo ""
echo "Done."
echo "Output files in: $WORKDIR"
echo " - %x_%j.out / %x_%j.err"
echo " - chain_omp_thread*.dat (if WRITE_CHAIN=1)"
