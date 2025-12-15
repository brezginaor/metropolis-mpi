#!/bin/bash
#SBATCH -p cascade
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=48
#SBATCH -t 01:30:00
#SBATCH -J metro_omp_sweep
#SBATCH -o /home/ipmmstudy2/tm5u6/metropolis-mpi/logs/omp_sweep_%j.out

set -euo pipefail

module load compiler/gcc/11

ROOT="$HOME/metropolis-mpi"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

cd "$ROOT/metropolis-omp"

echo "Date              = $(date)"
echo "Host              = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo ""

# --- СБОРКА ---
gcc -O3 -std=c99 -fopenmp metropolis_omp.c -o metropolis_omp -lm

TOTAL_T=10000000
SIGMA=0.5
REPS=3

OUT="$ROOT/c_omp_sweep_1_48_cascade_${SLURM_JOB_ID}.csv"
echo "p,elapsed" > "$OUT"

for NT in $(seq 1 48); do
  [ "$NT" -le "${SLURM_CPUS_PER_TASK}" ] || continue

  echo "Running OMP_NUM_THREADS=$NT..."

  for r in $(seq 1 $REPS); do
    export OMP_NUM_THREADS="$NT"
    line=$(srun ./metropolis_omp "$TOTAL_T" "$SIGMA" | tail -n 1)
    elapsed=$(echo "$line" | sed -n 's/.*elapsed=\([0-9.]*\).*/\1/p')
    [ -n "$elapsed" ] || { echo "Parse error (NT=$NT rep=$r): $line"; exit 1; }
    echo "  rep $r: $elapsed"
    echo "$elapsed" >> /tmp/times_${SLURM_JOB_ID}_${NT}.txt
  done

  med=$(sort -n /tmp/times_${SLURM_JOB_ID}_${NT}.txt | awk 'NR==2{print $1}')
  rm -f /tmp/times_${SLURM_JOB_ID}_${NT}.txt

  echo "$NT,$med" >> "$OUT"
done

echo ""
echo "Done. CSV saved to: $OUT"
