#!/bin/bash
#SBATCH --job-name=metro_py_mpi_sweep
#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH --partition=tornado
#SBATCH --output=/home/ipmmstudy2/tm5u6/metropolis-mpi/logs/py_mpi_sweep_%j.out

set -euo pipefail

# --- чтобы лог точно было куда писать ---
mkdir -p /home/ipmmstudy2/tm5u6/metropolis-mpi/logs

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo ""

module load mpi/openmpi/4.0.1/gcc/9
module load python/3.11

echo "Python path: $(which python3 || true)"
python3 -V

# Проверка mpi4py
python3 -c "import mpi4py; import numpy; print('mpi4py ok')"
echo ""

# --- переходим туда, где лежит твой python-код ---
cd "$HOME/metropolis-mpi/metropolis-py-mpi"

# !!! ВАЖНО: проверь имя файла !!!
# если файл называется metropolis_mpi.py -> оставь так
PYFILE="metropolis_mpi.py"
# если у тебя было metropolis_mpi.py (другая буква) — исправь здесь

TOTAL_T=200000
SIGMA=0.5
WRITE_CHAIN=0

OUT="$HOME/metropolis-mpi/py_mpi_sweep_1_112_tornado_${SLURM_JOB_ID}.csv"
echo "p,elapsed" > "$OUT"

for p in $(seq 1 112); do
  echo "Running p=$p..."

  # запускаем и берём последнюю строку (там у тебя печатается Elapsed time ...)
  line=$(mpirun -np $p --mca btl ^openib python3 "$PYFILE" "$TOTAL_T" "$SIGMA" "$WRITE_CHAIN" | tail -n 1)

  # парсим "Elapsed time (Python + MPI): X seconds"
  elapsed=$(echo "$line" | sed -n 's/.*Elapsed time (Python + MPI): \([0-9.]*\) seconds.*/\1/p')

  if [ -z "$elapsed" ]; then
    echo "Parse error. Last line was: $line"
    exit 1
  fi

  echo "$p,$elapsed" >> "$OUT"
done

echo ""
echo "Done. CSV saved to: $OUT"
