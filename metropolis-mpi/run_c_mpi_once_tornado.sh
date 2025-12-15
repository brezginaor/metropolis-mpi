#!/bin/bash
#SBATCH --job-name=metro_c_omp_once
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:20:00
#SBATCH --partition=tornado
#SBATCH --output=metro_c_omp_once_%j.out
#SBATCH --error=metro_c_omp_once_%j.err

set -euo pipefail

module purge
module load compiler/gcc/11

ROOT="$HOME/metropolis-mpi"
cd "$ROOT/metropolis-omp"

# без warning про rand_r
gcc -O3 -std=c99 -D_POSIX_C_SOURCE=200112L -fopenmp -DVERBOSE \
  metropolis_omp.c -o metropolis_omp -lm

TOTAL_T=10000000
SIGMA=0.5
OMP_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Nodes Allocated   = ${SLURM_JOB_NUM_NODES}"
echo "Tasks Allocated   = ${SLURM_NTASKS}"
echo "Cores/Task        = ${SLURM_CPUS_PER_TASK}"
echo ""
echo "Run config: OMP_NUM_THREADS=${OMP_THREADS} TOTAL_T=${TOTAL_T} SIGMA=${SIGMA}"
echo ""

export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

srun -n 1 -c "${SLURM_CPUS_PER_TASK}" ./metropolis_omp "$TOTAL_T" "$SIGMA"
