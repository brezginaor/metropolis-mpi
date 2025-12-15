#!/bin/bash
#SBATCH --job-name=metro_pth_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:30:00
#SBATCH --partition=tornado
#SBATCH --output=/home/ipmmstudy2/tm5u6/metropolis-mpi/logs/pth_sweep_%j.out

set -euo pipefail

ROOT="$HOME/metropolis-mpi"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

module load compiler/gcc/11

cd "$ROOT/metropolis_pthreads"

# Собираем
gcc -O3 -std=c99 metropolis_pthreads.c -o metropolis_pthreads \
  -I$HOME/sprng_lib/sprng/include \
  -L$HOME/sprng_lib/sprng/lib -llcg \
  -lpthread -lm

TOTAL_T=100000
SIGMA=0.5
WRITE_CHAIN=0
QUIET=1

REPS=3

OUT="$ROOT/c_pthreads_sweep_1_48_tornado_${SLURM_JOB_ID}.csv"
echo "p,elapsed" > "$OUT"

for p in $(seq 1 48); do
  echo "Running p=$p..."

  for r in $(seq 1 $REPS); do
    line=$(./metropolis_pthreads $TOTAL_T $SIGMA $p $WRITE_CHAIN $QUIET | tail -n 1)
    elapsed=$(echo "$line" | sed -n 's/.*elapsed=\([0-9.]*\).*/\1/p')
    if [ -z "$elapsed" ]; then
      echo "Parse error (p=$p, rep=$r). Last line was: $line"
      exit 1
    fi
    echo "  rep $r: $elapsed"
    echo "$elapsed" >> /tmp/times_${SLURM_JOB_ID}_${p}.txt
  done

  # медиана (для REPS=3 это 2-я строка после сортировки)
  med=$(sort -n /tmp/times_${SLURM_JOB_ID}_${p}.txt | awk 'NR==2{print $1}')
  rm -f /tmp/times_${SLURM_JOB_ID}_${p}.txt

  echo "$p,$med" >> "$OUT"
done

echo "Done: $OUT"
