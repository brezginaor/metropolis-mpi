#!/bin/bash
#SBATCH --job-name=metro_pth_once
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:30:00
#SBATCH --partition=tornado
#SBATCH --output=/home/ipmmstudy2/tm5u6/metropolis-mpi/metropolis_pthreads/metro_c_pth_once_%j.out

set -euo pipefail

ROOT="$HOME/metropolis-mpi"
WORKDIR="$ROOT/metropolis_pthreads"
cd "$WORKDIR"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Nodes Allocated   = ${SLURM_JOB_NUM_NODES}"
echo "Tasks Allocated   = ${SLURM_NTASKS}"
echo "Cores/Task        = ${SLURM_CPUS_PER_TASK}"
echo ""

module load compiler/gcc/11

# --- сборка ---
gcc -O3 -std=c99 metropolis_pthreads.c -o metropolis_pthreads \
  -I"$HOME/sprng_lib/sprng/include" \
  -L"$HOME/sprng_lib/sprng/lib" -llcg \
  -lpthread -lm

# --- число потоков из аргумента (как: sbatch run_pthreads_once.sh 32) ---
P="${1:-$SLURM_CPUS_PER_TASK}"
if ! [[ "$P" =~ ^[0-9]+$ ]]; then
  echo "Usage: sbatch $0 <P>"
  echo "P must be an integer."
  exit 1
fi
if [ "$P" -le 0 ]; then
  echo "P must be > 0"
  exit 1
fi
if [ "$P" -gt "$SLURM_CPUS_PER_TASK" ]; then
  echo "P=$P exceeds SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
  exit 1
fi

# --- параметры эксперимента ---
TOTAL_T=10000000
SIGMA=0.5
WRITE_CHAIN=0
QUIET=0   # важно: verbose вывод как в примере

echo "Run config: PTH_THREADS=${P} TOTAL_T=${TOTAL_T} SIGMA=${SIGMA} WRITE_CHAIN=${WRITE_CHAIN}"
echo ""

# --- запуск (лучше через srun для правильной привязки CPU) ---
srun ./metropolis_pthreads "$TOTAL_T" "$SIGMA" "$P" "$WRITE_CHAIN" "$QUIET"
