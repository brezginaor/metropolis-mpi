#!/bin/bash
#SBATCH --job-name=metro_py_mpi_one
#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --partition=tornado
#SBATCH --output=/home/ipmmstudy2/tm5u6/metropolis-mpi/metropolis-py-mpi/py_mpi_one_%j.out

set -euo pipefail

ROOT="$HOME/metropolis-mpi"
WORKDIR="$ROOT/metropolis-py-mpi"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Nodes Allocated   = $SLURM_JOB_NUM_NODES"
echo "Tasks Allocated   = $SLURM_NTASKS"
echo "Cores/Task        = $SLURM_CPUS_PER_TASK"
echo ""

module load mpi/openmpi/4.0.1/gcc/9
module load python/3.11
source $HOME/venvs/py311-mpi/bin/activate

echo "python3: $(which python3)"
python3 -V
python3 -c "import mpi4py; print('mpi4py ok')"
echo ""

cd "$WORKDIR"
PYFILE="metropolis_mpi.py"

# параметры эксперимента
TOTAL_T=10000000
SIGMA=0.5
WRITE_CHAIN=0
QUIET=1

# сколько ранков запускать
NP=112
REPS=3

OUT="$WORKDIR/py_mpi_one_np${NP}_${SLURM_JOB_ID}.csv"
echo "p,elapsed" > "$OUT"

echo "Running one case: NP=$NP (REPS=$REPS)..."

for r in $(seq 1 $REPS); do
  line=$(mpirun -np "$NP" --mca btl ^openib \
         python3 "$PYFILE" "$TOTAL_T" "$SIGMA" "$WRITE_CHAIN" "$QUIET" | tail -n 1)

  elapsed=$(echo "$line" | sed -n 's/.*elapsed=\([0-9.]*\).*/\1/p')
  if [ -z "$elapsed" ]; then
    echo "Parse error (rep=$r). Last line was: $line"
    exit 1
  fi

  echo "  rep $r: $elapsed"
  echo "$elapsed" >> /tmp/times_${SLURM_JOB_ID}_np${NP}.txt
done

med=$(sort -n /tmp/times_${SLURM_JOB_ID}_np${NP}.txt | awk 'NR==2{print $1}')
rm -f /tmp/times_${SLURM_JOB_ID}_np${NP}.txt

echo "$NP,$med" >> "$OUT"

echo ""
echo "Done. CSV saved to: $OUT"
