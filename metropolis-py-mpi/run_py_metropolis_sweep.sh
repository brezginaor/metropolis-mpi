#!/bin/bash
#SBATCH -p cascade
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 00:10:00
#SBATCH -J metro_py_sweep
#SBATCH -o metro_py_sweep_%j.out

module load compiler/gcc/11 mpi/openmpi/4.1.3/gcc/11 python/3.10

# один раз (не в скрипте) нужно будет сделать:
#   python -m pip install --user mpi4py

echo "Host: $(hostname)"
echo "SLURM_NTASKS = ${SLURM_NTASKS}"
echo

cd $HOME/metropolis-mpi/metropolis-py-mpi

TOTAL_T=200000
SIGMA=0.5

for NP in 1 2 4 8 16; do
    echo "=============================="
    echo "Running with NP = ${NP}"
    echo "=============================="
    mpirun -np ${NP} python metropolis_mpi.py ${TOTAL_T} ${SIGMA}
    echo
done
