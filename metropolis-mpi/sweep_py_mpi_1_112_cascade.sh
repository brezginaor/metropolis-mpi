#!/bin/bash
#SBATCH --job-name=metro_c_mpi_sweep
#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=1:30:00
#SBATCH --partition=cascade
#SBATCH --output=/home/ipmmstudy2/tm5u6/metropolis-mpi/logs/c_mpi_sweep_%j.out

set -euo pipefail

mkdir -p /home/ipmmstudy2/tm5u6/metropolis-mpi/logs

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo ""

module load compiler/gcc/11 mpi/openmpi/4.1.6/gcc/11

cd "$HOME/metropolis-mpi/metropolis-mpi"

INC="-I$HOME/sprng_lib/sprng/include"
LIB="-L$HOME/sprng_lib/sprng/lib -llcg -lm"

BIN="metropolic_${SLURM_JOB_ID}"
mpicc -O3 -std=c99 metropolic.c -o "$BIN" $INC $LIB

TOTAL_T=100000000
SIGMA=0.5
QUIET=1
REPS=3

OUT="$HOME/metropolis-mpi/c_mpi_sweep_1_112_tornado_${SLURM_JOB_ID}.csv"
echo "p,elapsed" > "$OUT"

for p in $(seq 1 112); do
  best=""
  for r in $(seq 1 $REPS); do
    line=$(mpirun -np $p --mca btl ^openib ./"$BIN" $TOTAL_T $SIGMA $QUIET | tail -n 1)
    elapsed=$(echo "$line" | sed -n 's/.*elapsed=\([0-9.]*\).*/\1/p')
    echo "  rep $r: $elapsed"
    echo "$elapsed" >> /tmp/times_${SLURM_JOB_ID}_${p}.txt
  done
  # медиана:
  med=$(sort -n /tmp/times_${SLURM_JOB_ID}_${p}.txt | awk 'NR==2{print $1}')
  rm -f /tmp/times_${SLURM_JOB_ID}_${p}.txt
  echo "$p,$med" >> "$OUT"
done

echo ""
echo "Done. CSV saved to: $OUT"
