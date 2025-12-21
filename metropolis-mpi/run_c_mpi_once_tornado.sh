#!/bin/bash
#SBATCH --job-name=metro_c_mpi_once
#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --partition=tornado
#SBATCH --output=%x_%j.out

set -euo pipefail

ROOT="$HOME/metropolis-mpi"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

# чтобы лог лежал в logs/
#exec >"$LOGDIR/metro_c_mpi_once_${SLURM_JOB_ID}.out" 2>&1

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Nodes Allocated   = ${SLURM_JOB_NUM_NODES}"
echo "Tasks Allocated   = ${SLURM_NTASKS}"
echo "Cores/Task        = ${SLURM_CPUS_PER_TASK}"
echo ""

module purge
module load compiler/gcc/11 mpi/openmpi/4.1.6/gcc/11

cd "$ROOT/metropolis-mpi"

INC="-I$HOME/sprng_lib/sprng/include"
LIB="-L$HOME/sprng_lib/sprng/lib -llcg -lm"

BIN="metropolic_${SLURM_JOB_ID}"
mpicc -O3 -std=c99 metropolic.c -o "$BIN" $INC $LIB

# параметры (можно поменять)
P="${1:-112}"          # сколько процессов запустить
TOTAL_T="${2:-100000000}"
SIGMA="${3:-0.5}"
WRITE_CHAIN=1
QUIET=0

echo "Run config: P=$P TOTAL_T=$TOTAL_T SIGMA=$SIGMA QUIET=$QUIET"
echo ""

mpirun -np "$P" --mca btl ^openib "./$BIN" "$TOTAL_T" "$SIGMA" "$WRITE_CHAIN" "$QUIET"

echo ""
echo "Done."
echo "Log: $LOGDIR/metro_c_mpi_once_${SLURM_JOB_ID}.out"
