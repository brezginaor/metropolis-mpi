#!/bin/bash
#SBATCH -p cascade                 # partition
#SBATCH -N 1                       # один узел
#SBATCH -n 1                       # один MPI-процесс (мы его не используем, но пусть будет)
#SBATCH --cpus-per-task=16         # до 16 потоков OpenMP
#SBATCH -t 00:10:00                # время заявки
#SBATCH -J metro_omp_sweep         # имя job'а
#SBATCH -o metro_omp_sweep_%j.out  # файл вывода

module load compiler/gcc/11

echo "Host: $(hostname)"
echo "SLURM_CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK}"
echo

# Переходим туда, где лежит бинарник OpenMP-версии
cd $HOME/metropolis-mpi/metropolis-omp

# Параметры алгоритма (общее число шагов по всем потокам вместе)
TOTAL_T=200000
SIGMA=0.5

for NT in 1 2 4 8 16; do
    echo "=============================="
    echo "Running with OMP_NUM_THREADS = ${NT}"
    echo "=============================="

    export OMP_NUM_THREADS=${NT}

    ./metropolis_omp ${TOTAL_T} ${SIGMA}

    echo
done
