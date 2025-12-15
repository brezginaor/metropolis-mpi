#!/bin/bash
#SBATCH --job-name=metro_py_mpi_sweep
#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --partition=tornado
#SBATCH --output=/home/ipmmstudy2/tm5u6/metropolis-mpi/logs/py_mpi_sweep_%j.out

set -euo pipefail

# куда писать логи и csv
ROOT="$HOME/metropolis-mpi"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

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

echo "python3: $(which python3)"
python3 -V
python3 -c "import mpi4py, numpy; print('mpi4py ok')"
echo ""

# переходим в папку с python-кодом
cd "$ROOT/metropolis-py-mpi"

PYFILE="metropolis_mpi.py"

# параметры эксперимента
TOTAL_T=10000000
SIGMA=0.5
WRITE_CHAIN=0
QUIET=1

OUT="$ROOT/py_mpi_sweep_1_112_tornado_${SLURM_JOB_ID}.csv"
echo "p,elapsed" > "$OUT"

for p in $(seq 1 112); do
  echo "Running p=$p..."

  # тихий режим: в конце должна быть строка "NP=... TOTAL_T=... elapsed=..."
  line=$(mpirun -np $p --mca btl ^openib python3 "$PYFILE" "$TOTAL_T" "$SIGMA" "$WRITE_CHAIN" "$QUIET" | tail -n 1)

  elapsed=$(echo "$line" | sed -n 's/.*elapsed=\([0-9.]*\).*/\1/p')
  if [ -z "$elapsed" ]; then
    echo "Parse error. Last line was: $line"
    exit 1
  fi

  echo "$p,$elapsed" >> "$OUT"
done

echo ""
echo "Done. CSV saved to: $OUT"
