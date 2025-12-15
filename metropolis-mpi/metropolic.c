#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "sprng.h"

#define SEED 123456789

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int *rng_stream = NULL;

void init_rng_stream(int rank, int size)
{
    rng_stream = init_rng(rank, size, SEED, 0);
    if (rng_stream == NULL) {
        fprintf(stderr, "Rank %d: init_rng failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void free_rng_stream(void)
{
    if (rng_stream != NULL) {
        free_rng(rng_stream);
        rng_stream = NULL;
    }
}

double rng_uniform01(void)
{
    return sprng(rng_stream);
}

double rng_normal01(void)
{
    double u1 = rng_uniform01();
    double u2 = rng_uniform01();

    if (u1 < 1e-12) u1 = 1e-12;

    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;

    return r * cos(theta);
}

double log_pi(const double x[2])
{
    return -0.5 * (x[0] * x[0] + x[1] * x[1]);
}

void q_sample(const double x[2], double y[2], double sigma)
{
    y[0] = x[0] + sigma * rng_normal01();
    y[1] = x[1] + sigma * rng_normal01();
}

void metropolis_step(double x[2], double *logp, double sigma, int *accepted)
{
    double y[2];
    q_sample(x, y, sigma);

    double logp_new = log_pi(y);
    double log_r = logp_new - (*logp);

    if (log_r >= 0.0) {
        x[0] = y[0];
        x[1] = y[1];
        *logp = logp_new;
        *accepted = 1;
    } else {
        double u = rng_uniform01();
        if (u < exp(log_r)) {
            x[0] = y[0];
            x[1] = y[1];
            *logp = logp_new;
            *accepted = 1;
        } else {
            *accepted = 0;
        }
    }
}

/* одна цепочка */
double run_chain(const double x0[2], double sigma, int T, int rank,
                 int write_chain,
                 int *accepted_total_out,
                 double sum_x_out[2],
                 double sumsq_x_out[2])
{
    double x[2] = {x0[0], x0[1]};
    double logp = log_pi(x);

    int accepted_count = 0;
    int acc;

    double sum_x[2]   = {0.0, 0.0};
    double sumsq_x[2] = {0.0, 0.0};

    FILE *f = NULL;
    if (write_chain) {
        char filename[64];
        snprintf(filename, sizeof(filename), "chain_rank%d.dat", rank);
        f = fopen(filename, "w");
        if (!f) {
            fprintf(stderr, "Rank %d: cannot open output file %s\n", rank, filename);
            return -1.0;
        }
        fprintf(f, "# t x0 x1\n");
        fprintf(f, "0 %.10f %.10f\n", x[0], x[1]);
    }

    for (int t = 1; t < T; ++t) {
        metropolis_step(x, &logp, sigma, &acc);
        accepted_count += acc;

        sum_x[0]   += x[0];
        sum_x[1]   += x[1];
        sumsq_x[0] += x[0] * x[0];
        sumsq_x[1] += x[1] * x[1];

        if (f) {
            fprintf(f, "%d %.10f %.10f\n", t, x[0], x[1]);
        }
    }

    if (f) fclose(f);

    if (accepted_total_out) *accepted_total_out = accepted_count;
    if (sum_x_out)   { sum_x_out[0] = sum_x[0];   sum_x_out[1] = sum_x[1]; }
    if (sumsq_x_out) { sumsq_x_out[0] = sumsq_x[0]; sumsq_x_out[1] = sumsq_x[1]; }

    double acc_rate = (T > 1) ? ((double)accepted_count / (double)(T - 1)) : 0.0;
    return acc_rate;
}

int main(int argc, char *argv[])
{
    int rank, size;

    int    TOTAL_T     = 200000;
    double sigma       = 0.5;
    double x0[2]       = {0.0, 0.0};
    int    write_chain = 0;
    int    quiet       = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* argv: TOTAL_T sigma write_chain quiet */
    if (argc > 1) { int tmpT = atoi(argv[1]); if (tmpT > 0) TOTAL_T = tmpT; }
    if (argc > 2) { double tmpS = atof(argv[2]); if (tmpS > 0.0) sigma = tmpS; }
    if (argc > 3) { write_chain = (atoi(argv[3]) != 0); }
    if (argc > 4) { quiet = (atoi(argv[4]) != 0); }

    int base_T = TOTAL_T / size;
    int rem    = TOTAL_T % size;
    int T_local = (rank < rem) ? (base_T + 1) : base_T;
    if (T_local < 2) T_local = 2;

    double t_start = MPI_Wtime();

    if (rank == 0 && !quiet) {
        printf("Metropolis + MPI + SPRNG\n");
        printf("Processes: %d\n", size);
        printf("TOTAL_T (sum over ranks): %d\n", TOTAL_T);
        printf("Local T on rank 0: %d\n", T_local);
        printf("Proposal sigma: %f\n", sigma);
        printf("WRITE_CHAIN: %d\n\n", write_chain);
    }

    init_rng_stream(rank, size);

    int    local_accepted   = 0;
    double local_sum_x[2]   = {0.0, 0.0};
    double local_sumsq_x[2] = {0.0, 0.0};

    double local_acc_rate = run_chain(x0, sigma, T_local, rank, write_chain,
                                      &local_accepted, local_sum_x, local_sumsq_x);

    /* acceptance per rank */
    double *all_rates = NULL;
    if (rank == 0) all_rates = (double *)malloc(size * sizeof(double));

    MPI_Gather(&local_acc_rate, 1, MPI_DOUBLE,
               all_rates,        1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0 && !quiet) {
        printf("Acceptance rates per rank:\n");
        double sum_rates = 0.0;
        for (int i = 0; i < size; ++i) {
            printf("  rank %d: %.4f\n", i, all_rates[i]);
            sum_rates += all_rates[i];
        }
        printf("Average acceptance rate (by ranks): %.4f\n\n", sum_rates / (double)size);
    }
    if (all_rates) free(all_rates);

    /* global sums via Reduce (проще и быстрее, чем ручной Send/Recv) */
    int    global_accepted = 0;
    double global_sum_x[2] = {0.0, 0.0};
    double global_sumsq_x[2] = {0.0, 0.0};
    long long local_samples = (T_local > 1) ? (long long)(T_local - 1) : 0;
    long long global_samples = 0;

    MPI_Reduce(&local_accepted, &global_accepted, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_sum_x, global_sum_x, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_sumsq_x, global_sumsq_x, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_samples, &global_samples, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    if (rank == 0) {
        if (quiet) {
            printf("NP=%d TOTAL_T=%d elapsed=%.6f\n", size, TOTAL_T, elapsed);
        } else {
            long long N = (global_samples > 0) ? global_samples : 1;

            double mean0 = global_sum_x[0] / (double)N;
            double mean1 = global_sum_x[1] / (double)N;

            double var0 = global_sumsq_x[0] / (double)N - mean0 * mean0;
            double var1 = global_sumsq_x[1] / (double)N - mean1 * mean1;

            double acc_rate_global = (double)global_accepted / (double)N;

            printf("Global stats over all ranks:\n");
            printf("  Samples used (sum over ranks): %lld\n", N);
            printf("  Acceptance rate (per step): %.6f\n", acc_rate_global);
            printf("  Mean:     [%.6f, %.6f]\n", mean0, mean1);
            printf("  Variance: [%.6f, %.6f]\n", var0, var1);
            printf("\nElapsed time: %.6f seconds\n", elapsed);
        }
    }

    free_rng_stream();
    MPI_Finalize();
    return 0;
}
