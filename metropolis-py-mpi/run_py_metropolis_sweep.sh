#!/bin/bash
#SBATCH --job-name=metro_py_mpi_sweep
#SBATCH --nodes=4
#SBATCH --ntasks=112
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=cascade
#SBATCH --output=logs/py_mpi_sweep_cascade_%j.out

set -e

module load mpi/openmpi/4.0.1/gcc/9
module load python/3.11

# Быстрая проверка, что mpi4py есть
python3 -c "import mpi4py; print('mpi4py ok')" 

cd "$HOME/metropolis-mpi/metropolis-py-mpi"   # <- если у тебя python-код лежит тут
# если файл лежит в другом месте — поправь путь и имя

TOTAL_T=200000
SIGMA=0.5
WRITE_CHAIN=0

OUT="$HOME/metropolis-mpi/py_mpi_sweep_1_112_cascade_${SLURM_JOB_ID}.csv"
echo "p,elapsed" > "$OUT"

for p in $(seq 1 112); do
  echo "Running p=$p..."

  line=$(mpirun -np $p --mca btl ^openib python3 metropolis_mpi.py $TOTAL_T $SIGMA $WRITE_CHAIN | tail -n 1)


  elapsed=$(echo "$line" | sed -n 's/.*Elapsed time (Python + MPI): \([0-9.]*\).*/\1/p')



  [ -n "$elapsed" ] || { echo "Parse error: $line"; exit 1; }
  echo "$p,$elapsed" >> "$OUT"
done

echo "Done: $OUT"