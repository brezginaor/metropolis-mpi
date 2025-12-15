#!/usr/bin/env python3
import sys
import math
from mpi4py import MPI
import numpy as np

SEED = 123456789


# ---------- RNG на каждом ранге ----------

def init_rng(rank: int) -> np.random.Generator:
    """
    Создаём независимый генератор NumPy для каждого ранга.
    """
    seed_seq = np.random.SeedSequence(SEED + rank)
    return np.random.default_rng(seed_seq)


def rng_uniform01(rng: np.random.Generator) -> float:
    return float(rng.random())


def rng_normal01(rng: np.random.Generator) -> float:
    return float(rng.normal())


# ---------- целевая плотность π(x) ~ N(0, I) в 2D ----------

def log_pi(x: np.ndarray) -> float:
    """
    x — numpy-вектор длины 2.
    log π(x) для стандартной 2D нормали (с точностью до константы).
    """
    return -0.5 * float(np.dot(x, x))


# ---------- предложение q: y = x + N(0, sigma^2 I) ----------

def q_sample(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    x — numpy-вектор длины 2.
    Возвращает новое предложение y.
    """
    return x + sigma * rng.normal(size=2)


def metropolis_step(x: np.ndarray, logp: float, sigma: float, rng: np.random.Generator):
    """
    Один шаг алгоритма Метрополиса.
    Возвращает (x_new, logp_new, accepted).
    """
    y = q_sample(x, sigma, rng)
    logp_new = log_pi(y)
    log_r = logp_new - logp

    if log_r >= 0.0:
        return y, logp_new, 1

    u = rng_uniform01(rng)
    if u < math.exp(log_r):
        return y, logp_new, 1
    return x, logp, 0


# ---------- одна локальная цепочка на ранге ----------

def run_chain(
    x0: np.ndarray,
    sigma: float,
    T_local: int,
    rank: int,
    rng: np.random.Generator,
    write_chain: int,
):
    """
    Запускает одну цепочку длины T_local на данном ранге.

    Если write_chain != 0, пишет chain_rank<rank>.dat.
    Возвращает:
      acc_rate, accepted_total, sum_x(2), sumsq_x(2), samples_used
    """
    x = np.array(x0, dtype=float)
    logp = log_pi(x)

    accepted_total = 0
    sum_x = np.zeros(2, dtype=float)
    sumsq_x = np.zeros(2, dtype=float)

    f = None
    if write_chain:
        filename = f"chain_rank{rank}.dat"
        f = open(filename, "w", encoding="utf-8")
        f.write("# t x0 x1\n")
        f.write(f"0 {x[0]:.10f} {x[1]:.10f}\n")

    # статистику считаем по t=1..T_local-1
    for t in range(1, T_local):
        x, logp, acc = metropolis_step(x, logp, sigma, rng)
        accepted_total += acc

        sum_x += x
        sumsq_x += x * x

        if f is not None:
            f.write(f"{t} {x[0]:.10f} {x[1]:.10f}\n")

    if f is not None:
        f.close()

    samples_used = max(T_local - 1, 0)
    acc_rate = accepted_total / samples_used if samples_used > 0 else 0.0
    return acc_rate, accepted_total, sum_x, sumsq_x, samples_used


# ---------- main с MPI ----------

def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # параметры по умолчанию
    TOTAL_T = 200_000     # суммарное число шагов по всем процессам
    sigma = 0.5
    x0 = np.array([0.0, 0.0], dtype=float)
    WRITE_CHAIN = 0       # 0 = не писать chain_rank*.dat (важно для свипов)

    # argv: TOTAL_T sigma write_chain
    if len(sys.argv) > 1:
        tmpT = int(sys.argv[1])
        if tmpT > 0:
            TOTAL_T = tmpT
    if len(sys.argv) > 2:
        tmpS = float(sys.argv[2])
        if tmpS > 0.0:
            sigma = tmpS
    if len(sys.argv) > 3:
        WRITE_CHAIN = int(sys.argv[3])

    # распределяем TOTAL_T по ранкам
    base_T = TOTAL_T // size
    remainder = TOTAL_T % size
    T_local = base_T + 1 if rank < remainder else base_T
    if T_local < 2:
        T_local = 2

    t_start = MPI.Wtime()

    if rank == 0:
        print("Metropolis + Python + MPI")
        print(f"Processes: {size}")
        print(f"TOTAL_T (sum over ranks): {TOTAL_T}")
        print(f"Local T on rank 0: {T_local}")
        print(f"Proposal sigma: {sigma:.6f}")
        print(f"WRITE_CHAIN: {WRITE_CHAIN}")
        print()

    rng = init_rng(rank)

    (local_acc_rate,
     local_accepted,
     local_sum_x,
     local_sumsq_x,
     local_samples_used) = run_chain(
        x0, sigma, T_local, rank, rng, WRITE_CHAIN
    )

    # acceptance rate по ранкам
    all_rates = comm.gather(local_acc_rate, root=0)

    if rank == 0:
        print("Acceptance rates per rank:")
        avg_rate = 0.0
        for r, rate in enumerate(all_rates):
            print(f"  rank {r}: {rate:.4f}")
            avg_rate += rate
        avg_rate /= size
        print(f"Average acceptance rate (by ranks): {avg_rate:.4f}")
        print()

    # глобальные суммы
    global_accepted = comm.reduce(local_accepted, op=MPI.SUM, root=0)
    global_sum_x = comm.reduce(local_sum_x, op=MPI.SUM, root=0)
    global_sumsq_x = comm.reduce(local_sumsq_x, op=MPI.SUM, root=0)
    global_samples_used = comm.reduce(local_samples_used, op=MPI.SUM, root=0)

    t_end = MPI.Wtime()

    if rank == 0:
        N = global_samples_used if global_samples_used > 0 else 1
        mean = global_sum_x / N
        var = global_sumsq_x / N - mean * mean
        acc_rate_global = global_accepted / N

        print("Global stats over all ranks:")
        print(f"  Samples used (sum over ranks): {N}")
        print(f"  Acceptance rate (per step): {acc_rate_global:.6f}")
        print(f"  Mean:     [{mean[0]:.6f}, {mean[1]:.6f}]")
        print(f"  Variance: [{var[0]:.6f}, {var[1]:.6f}]")
        print()
        print(f"Elapsed time (Python + MPI): {t_end - t_start:.6f} seconds")


if __name__ == "__main__":
    main()
