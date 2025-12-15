#!/bin/bash
#SBATCH --job-name=metro_c_omp_once
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:20:00
#SBATCH --partition=tornado
#SBATCH --output=/home/ipmmstudy2/tm5u6/metropolis-mpi/logs/metro_c_omp_once_%j.out

set -euo pipefail

ROOT="$HOME/metropolis-mpi"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Nodes Allocated   = ${SLURM_JOB_NUM_NODES}"
echo "Tasks Allocated   = ${SLURM_NTASKS}"
echo "Cores/Task        = ${SLURM_CPUS_PER_TASK}"
echo ""

module purge
module load compiler/gcc/11

cd "$ROOT/metropolis-omp"

# --- СБОРКА ---
# ВАЖНО: чтобы печатались ВСЕ статистики, собираем с -DVERBOSE
gcc -O3 -std=c99 -fopenmp -DVERBOSE metropolis_omp.c -o metropolis_omp -lm

# --- ПАРАМЕТРЫ ОДНОГО ЗАПУСКА ---
TOTAL_T=10000000
SIGMA=0.5
OMP_THREADS=48          # сколько потоков хотим
WRITE_CHAIN=0           # у тебя в коде сейчас нет аргумента, просто для ясности

export OMP_NUM_THREADS="$OMP_THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Run config: OMP_NUM_THREADS=${OMP_NUM_THREADS} TOTAL_T=${TOTAL_T} SIGMA=${SIGMA}"
echo ""

# Запуск через srun (правильнее для SLURM)
srun -n 1 -c "${SLURM_CPUS_PER_TASK}" ./metropolis_omp "$TOTAL_T" "$SIGMA"

echo ""
echo "Done."
echo "Log: ${LOGDIR}/metro_c_omp_once_${SLURM_JOB_ID}.out"
