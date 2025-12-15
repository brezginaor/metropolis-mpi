#!/bin/bash
#SBATCH --job-name=metro_c_mpi_sweep
#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --partition=tornado
#SBATCH --output=/home/ipmmstudy2/tm5u6/metropolis-mpi/logs/c_mpi_sweep_%j.out

set -euo pipefail

ROOT="$HOME/metropolis-mpi"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo ""

module load compiler/gcc/11 mpi/openmpi/4.1.6/gcc/11

cd "$ROOT/metropolis-mpi"

INC="-I$HOME/sprng_lib/sprng/include"
LIB="-L$HOME/sprng_lib/sprng/lib -llcg -lm"

BIN="metropolic_${SLURM_JOB_ID}"
mpicc -O3 -std=c99 metropolic.c -o "$BIN" $INC $LIB

TOTAL_T=100000000
SIGMA=0.5
QUIET=1
REPS=3

OUT="$ROOT/c_mpi_sweep_1_112_tornado_${SLURM_JOB_ID}.csv"
echo "p,elapsed" > "$OUT"

for p in $(seq 1 112); do
  echo "Running p=$p..."

  tmp="/tmp/times_${SLURM_JOB_ID}_${p}.txt"
  : > "$tmp"

  for r in $(seq 1 "$REPS"); do
    line=$(mpirun -np "$p" --mca btl ^openib ./"$BIN" "$TOTAL_T" "$SIGMA" "$QUIET" | tail -n 1)
    elapsed=$(echo "$line" | sed -n 's/.*elapsed=\([0-9.]*\).*/\1/p')

    if [ -z "$elapsed" ]; then
      echo "Parse error. Last line was: $line"
      exit 1
    fi

    echo "  rep $r: $elapsed"
    echo "$elapsed" >> "$tmp"
  done

  # медиана по REPS значениям (для чётного REPS — среднее двух центральных)
  med=$(
    sort -n "$tmp" | awk '
      {a[NR]=$1}
      END{
        n=NR
        if(n%2==1) { print a[(n+1)/2] }
        else       { print (a[n/2] + a[n/2+1]) / 2.0 }
      }'
  )

  rm -f "$tmp"
  echo "$p,$med" >> "$OUT"
done

echo ""
echo "Done. CSV saved to: $OUT"
