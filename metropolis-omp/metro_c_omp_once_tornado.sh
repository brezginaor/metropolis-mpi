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

TOTAL_T=1000000   # Общее количество шагов
SIGMA=0.5           # Стандартное отклонение для предложения
OMP_THREADS="${SLURM_CPUS_PER_TASK}"  # Количество потоков, как в SLURM

# Параметры записи (по умолчанию каждый шаг)
WRITE_CHAIN=1       # 0 - не записывать, 1 - записывать
WRITE_EVERY=1000    # Записывать каждую 1000-ю точку (по умолчанию)

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Nodes Allocated   = ${SLURM_JOB_NUM_NODES}"
echo "Tasks Allocated   = ${SLURM_NTASKS}"
echo "Cores/Task        = ${SLURM_CPUS_PER_TASK}"
echo ""
echo "Run config: OMP_NUM_THREADS=${OMP_THREADS} TOTAL_T=${TOTAL_T} SIGMA=${SIGMA} WRITE_CHAIN=${WRITE_CHAIN} WRITE_EVERY=${WRITE_EVERY}"
echo ""

# Запуск программы с передачей параметров в программу
srun -n 1 -c "${SLURM_CPUS_PER_TASK}" ./metropolis_omp "$TOTAL_T" "$SIGMA" "$WRITE_CHAIN" "$WRITE_EVERY"
