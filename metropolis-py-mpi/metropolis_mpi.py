#!/usr/bin/env python3
import sys
import math
from mpi4py import MPI
import random

SEED = 123456789


def init_rng(rank: int) -> random.Random:
    return random.Random(SEED + rank)


def rng_uniform01(rng: random.Random) -> float:
    return rng.random()


def rng_normal01(rng: random.Random) -> float:
    return rng.gauss(0.0, 1.0)


def log_pi(x0: float, x1: float) -> float:
    return -0.5 * (x0 * x0 + x1 * x1)


def q_sample(x0: float, x1: float, sigma: float, rng: random.Random):
    y0 = x0 + sigma * rng_normal01(rng)
    y1 = x1 + sigma * rng_normal01(rng)
    return y0, y1


def metropolis_step(x0: float, x1: float, logp: float, sigma: float, rng: random.Random):
    y0, y1 = q_sample(x0, x1, sigma, rng)
    logp_new = log_pi(y0, y1)
    log_r = logp_new - logp

    if log_r >= 0.0:
        return y0, y1, logp_new, 1

    u = rng_uniform01(rng)
    if u < math.exp(log_r):
        return y0, y1, logp_new, 1

    return x0, x1, logp, 0


def run_chain_compute_only(x0_init: float, x1_init: float,
                           sigma: float, T_local: int, rng: random.Random,
                           need_chain: int):
    """
    Вычислительная часть (без записи в файл).
    Если need_chain=1 -> сохраняем траекторию в память и вернём её, чтобы записать после таймера.
    """
    x0 = x0_init
    x1 = x1_init
    logp = log_pi(x0, x1)

    accepted_total = 0
    sum_x0 = 0.0
    sum_x1 = 0.0
    sumsq_x0 = 0.0
    sumsq_x1 = 0.0

    chain = None
    if need_chain:
        chain = [(0, x0, x1)]

    t0 = MPI.Wtime()

    for t in range(1, T_local):
        x0, x1, logp, acc = metropolis_step(x0, x1, logp, sigma, rng)
        accepted_total += acc

        sum_x0 += x0
        sum_x1 += x1
        sumsq_x0 += x0 * x0
        sumsq_x1 += x1 * x1

        if chain is not None:
            chain.append((t, x0, x1))

    t1 = MPI.Wtime()
    compute_time = t1 - t0

    samples_used = max(T_local - 1, 0)
    acc_rate = accepted_total / samples_used if samples_used > 0 else 0.0

    return (acc_rate, accepted_total, sum_x0, sum_x1, sumsq_x0, sumsq_x1,
            samples_used, compute_time, chain)


def write_chain_file(rank: int, chain):
    with open(f"chain_rank{rank}.dat", "w", encoding="utf-8") as f:
        f.write("# t x0 x1\n")
        for t, x0, x1 in chain:
            f.write(f"{t} {x0:.10f} {x1:.10f}\n")


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    TOTAL_T = 200_000
    sigma = 0.5
    WRITE_CHAIN = 0
    QUIET = 0

    # argv: TOTAL_T sigma write_chain quiet
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
    if len(sys.argv) > 4:
        QUIET = 1 if int(sys.argv[4]) != 0 else 0

    # Раздаём TOTAL_T по ранкам
    base_T = TOTAL_T // size
    rem = TOTAL_T % size
    T_local = base_T + 1 if rank < rem else base_T
    if T_local < 2:
        T_local = 2

    if rank == 0 and not QUIET:
        print("Metropolis + Python + MPI (no numpy)")
        print(f"Processes: {size}")
        print(f"TOTAL_T (sum over ranks): {TOTAL_T}")
        print(f"Local T on rank 0: {T_local}")
        print(f"Proposal sigma: {sigma:.6f}")
        print(f"WRITE_CHAIN: {WRITE_CHAIN}")
        print()

    rng = init_rng(rank)

    (local_acc_rate, local_accepted,
     local_sum_x0, local_sum_x1,
     local_sumsq_x0, local_sumsq_x1,
     local_samples, local_compute,
     chain) = run_chain_compute_only(
        0.0, 0.0, sigma, T_local, rng, WRITE_CHAIN
    )

    # Время параллельной вычислительной части = максимум по ранкам
    elapsed_compute = comm.reduce(local_compute, op=MPI.MAX, root=0)

    all_rates = comm.gather(local_acc_rate, root=0)

    if rank == 0 and not QUIET:
        print("Acceptance rates per rank:")
        avg = 0.0
        for r, rate in enumerate(all_rates):
            print(f"  rank {r}: {rate:.4f}")
            avg += rate
        avg /= size
        print(f"Average acceptance rate (by ranks): {avg:.4f}")
        print()

    # Reduce статистик
    global_accepted = comm.reduce(local_accepted, op=MPI.SUM, root=0)
    global_sum_x0 = comm.reduce(local_sum_x0, op=MPI.SUM, root=0)
    global_sum_x1 = comm.reduce(local_sum_x1, op=MPI.SUM, root=0)
    global_sumsq_x0 = comm.reduce(local_sumsq_x0, op=MPI.SUM, root=0)
    global_sumsq_x1 = comm.reduce(local_sumsq_x1, op=MPI.SUM, root=0)
    global_samples = comm.reduce(local_samples, op=MPI.SUM, root=0)

    # Запись цепочки ПОСЛЕ измерения (не входит в elapsed_compute)
    if WRITE_CHAIN and chain is not None:
        write_chain_file(rank, chain)

    if rank == 0:
        if QUIET:
            print(f"NP={size} TOTAL_T={TOTAL_T} elapsed={elapsed_compute:.6f}")
            return

        N = global_samples if global_samples > 0 else 1
        mean0 = global_sum_x0 / N
        mean1 = global_sum_x1 / N
        var0 = global_sumsq_x0 / N - mean0 * mean0
        var1 = global_sumsq_x1 / N - mean1 * mean1
        acc_rate_global = global_accepted / N

        print("Global stats over all ranks:")
        print(f"  Samples used (sum over ranks): {N}")
        print(f"  Acceptance rate (per step): {acc_rate_global:.6f}")
        print(f"  Mean:     [{mean0:.6f}, {mean1:.6f}]")
        print(f"  Variance: [{var0:.6f}, {var1:.6f}]")
        print()
        print(f"Elapsed time (Python + MPI, compute-only): {elapsed_compute:.6f} seconds")


if __name__ == "__main__":
    main()
