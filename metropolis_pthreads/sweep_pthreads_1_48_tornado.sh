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

cd "$ROOT/metropolis-pthreads"

# Собираем
gcc -O3 -std=c99 metropolis_pthreads.c -o metropolis_pthreads \
  -I$HOME/sprng_lib/sprng/include \
  -L$HOME/sprng_lib/sprng/lib -llcg \
  -lpthread -lm

TOTAL_T=10000000
SIGMA=0.5
WRITE_CHAIN=0
QUIET=1

OUT="$ROOT/c_pthreads_sweep_1_48_tornado_${SLURM_JOB_ID}.csv"
echo "p,elapsed" > "$OUT"

for p in $(seq 1 48); do
  echo "Running p=$p..."
  line=$(./metropolis_pthreads $TOTAL_T $SIGMA $p $WRITE_CHAIN $QUIET | tail -n 1)
  elapsed=$(echo "$line" | sed -n 's/.*elapsed=\([0-9.]*\).*/\1/p')
  [ -n "$elapsed" ] || { echo "Parse error: $line"; exit 1; }
  echo "$p,$elapsed" >> "$OUT"
done

echo "Done: $OUT"
