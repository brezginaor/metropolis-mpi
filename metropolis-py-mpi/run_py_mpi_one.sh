#!/bin/bash
#SBATCH --job-name=metro_py_mpi_once
#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --partition=tornado
#SBATCH --output=/home/ipmmstudy2/tm5u6/metropolis-mpi/metropolis-py-mpi/metro_py_mpi_once_%j.out

set -euo pipefail

ROOT="$HOME/metropolis-mpi"
WORKDIR="$ROOT/metropolis-py-mpi"
cd "$WORKDIR"

# ---------- шапка лога как у тебя ----------
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Nodes Allocated   = ${SLURM_JOB_NUM_NODES}"
echo "Tasks Allocated   = ${SLURM_NTASKS}"
echo "Cores/Task        = ${SLURM_CPUS_PER_TASK}"
echo ""

# ---------- модули / окружение ----------
module load mpi/openmpi/4.0.1/gcc/9
module load python/3.11
source "$HOME/venvs/py311-mpi/bin/activate"

# ---------- аргумент NP ----------
NP="${1:-$SLURM_NTASKS}"
if ! [[ "$NP" =~ ^[0-9]+$ ]]; then
  echo "Usage: sbatch $0 <NP>"
  echo "NP must be an integer."
  exit 1
fi
if [ "$NP" -le 0 ]; then
  echo "NP must be > 0"
  exit 1
fi
if [ "$NP" -gt "$SLURM_NTASKS" ]; then
  echo "NP=$NP exceeds allocation SLURM_NTASKS=$SLURM_NTASKS"
  exit 1
fi

# ---------- параметры эксперимента ----------
PYFILE="metropolis_mpi.py"
TOTAL_T=10000000
SIGMA=0.5
WRITE_CHAIN=0
QUIET=0  

echo "Run config: NP=${NP} TOTAL_T=${TOTAL_T} SIGMA=${SIGMA} WRITE_CHAIN=${WRITE_CHAIN}"
echo ""

# ---------- запуск ----------
mpirun -np "$NP" --mca btl ^openib \
  python3 "$PYFILE" "$TOTAL_T" "$SIGMA" "$WRITE_CHAIN" "$QUIET"

