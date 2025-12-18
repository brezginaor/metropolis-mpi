#!/bin/bash
#SBATCH -p cascade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 01:30:00
#SBATCH -J metro_omp_sweep
#SBATCH -o /home/ipmmstudy2/tm5u6/metropolis-mpi/logs/omp_sweep_%j.out

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

module load compiler/gcc/11

cd "$ROOT/metropolis-omp"

# --- СБОРКА ---
gcc -O3 -std=c99 -fopenmp metropolis_omp.c -o metropolis_omp -lm

# --- параметры эксперимента ---
TOTAL_T=100000000
SIGMA=0.5
WRITE_CHAIN=0   # важно: 0 для свипов (не создаём файлы)
REPS=3

OUT="$ROOT/c_omp_sweep_1_48_cascade_${SLURM_JOB_ID}.csv"
echo "p,elapsed" > "$OUT"

for NT in $(seq 1 48); do
  if [ "$NT" -gt "${SLURM_CPUS_PER_TASK}" ]; then
    echo "Skipping NT=${NT}: exceeds SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
    continue
  fi

  echo "Running OMP_NUM_THREADS=$NT..."

  for r in $(seq 1 $REPS); do
    export OMP_NUM_THREADS="$NT"

    line=$(srun ./metropolis_omp "$TOTAL_T" "$SIGMA" "$WRITE_CHAIN" | tail -n 1)
    elapsed=$(echo "$line" | sed -n 's/.*elapsed=\([0-9.]*\).*/\1/p')

    if [ -z "$elapsed" ]; then
      echo "Parse error (NT=$NT, rep=$r). Last line was: $line"
      exit 1
    fi

    echo "  rep $r: $elapsed"
    echo "$elapsed" >> /tmp/times_${SLURM_JOB_ID}_${NT}.txt
  done

  # медиана (для REPS=3 это 2-я строка после сортировки)
  med=$(sort -n /tmp/times_${SLURM_JOB_ID}_${NT}.txt | awk 'NR==2{print $1}')
  rm -f /tmp/times_${SLURM_JOB_ID}_${NT}.txt

  echo "$NT,$med" >> "$OUT"
done

echo ""
echo "Done. CSV saved to: $OUT"
