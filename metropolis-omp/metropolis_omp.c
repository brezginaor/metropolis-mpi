#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------- RNG на потоке: rand_r + Бокс–Мюллер ---------- */

static inline double rng_uniform01(unsigned int *state)
{
    return (double)rand_r(state) / ((double)RAND_MAX + 1.0);
}

static inline double rng_normal01(unsigned int *state)
{
    double u1 = rng_uniform01(state);
    double u2 = rng_uniform01(state);

    if (u1 < 1e-12) u1 = 1e-12;

    double r     = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;

    return r * cos(theta);
}

/* ---------- целевая плотность π(x) ~ N(0, I) в 2D ---------- */

static inline double log_pi(const double x[2])
{
    return -0.5 * (x[0] * x[0] + x[1] * x[1]);
}

/* предложение: y = x + N(0, sigma^2 I) */
static inline void q_sample(const double x[2], double y[2], double sigma, unsigned int *rng_state)
{
    y[0] = x[0] + sigma * rng_normal01(rng_state);
    y[1] = x[1] + sigma * rng_normal01(rng_state);
}

/* один шаг Метрополиса */
static inline void metropolis_step(double x[2],
                                   double *logp,
                                   double sigma,
                                   int *accepted,
                                   unsigned int *rng_state)
{
    double y[2];
    q_sample(x, y, sigma, rng_state);

    double logp_new = log_pi(y);
    double log_r    = logp_new - (*logp);

    if (log_r >= 0.0) {
        x[0] = y[0];
        x[1] = y[1];
        *logp = logp_new;
        *accepted = 1;
        return;
    }

    double u = rng_uniform01(rng_state);
    if (u < exp(log_r)) {
        x[0] = y[0];
        x[1] = y[1];
        *logp = logp_new;
        *accepted = 1;
    } else {
        *accepted = 0;
    }
}

/* ---------- одна цепочка на одном потоке (вычисления) ---------- */
/*
  Если chain_out != NULL, то сохраняем траекторию:
    chain_out[2*t + 0] = x0(t)
    chain_out[2*t + 1] = x1(t)
  Это нужно, чтобы записать в файл ПОСЛЕ измерения времени.
*/
static double run_chain_compute(int T,
                                double sigma,
                                const double x0_init[2],
                                unsigned int *rng_state,
                                int *accepted_total_out,
                                double sum_x_out[2],
                                double sumsq_x_out[2],
                                double *chain_out /* nullable */)
{
    double x[2] = {x0_init[0], x0_init[1]};
    double logp = log_pi(x);

    int accepted_count = 0;

    double sum_x[2]   = {0.0, 0.0};
    double sumsq_x[2] = {0.0, 0.0};

    if (chain_out) {
        chain_out[0] = x[0];
        chain_out[1] = x[1];
    }

    for (int t = 1; t < T; ++t) {
        int acc = 0;
        metropolis_step(x, &logp, sigma, &acc, rng_state);
        accepted_count += acc;

        sum_x[0]   += x[0];
        sum_x[1]   += x[1];
        sumsq_x[0] += x[0] * x[0];
        sumsq_x[1] += x[1] * x[1];

        if (chain_out) {
            chain_out[2*t + 0] = x[0];
            chain_out[2*t + 1] = x[1];
        }
    }

    if (accepted_total_out) *accepted_total_out = accepted_count;
    if (sum_x_out)   { sum_x_out[0] = sum_x[0];   sum_x_out[1] = sum_x[1]; }
    if (sumsq_x_out) { sumsq_x_out[0] = sumsq_x[0]; sumsq_x_out[1] = sumsq_x[1]; }

    return (T > 1) ? ((double)accepted_count / (double)(T - 1)) : 0.0;
}

static void write_chain_file(int thread_id, int T, const double *chain)
{
    char filename[64];
    snprintf(filename, sizeof(filename), "chain_omp_thread%d.dat", thread_id);

    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Thread %d: cannot open output file %s\n", thread_id, filename);
        return;
    }

    fprintf(f, "# t x0 x1\n");
    for (int t = 0; t < T; ++t) {
        fprintf(f, "%d %.10f %.10f\n", t, chain[2*t + 0], chain[2*t + 1]);
    }
    fclose(f);
}

/* -------------------------- main с OpenMP -------------------------- */

int main(int argc, char *argv[])
{
    long long TOTAL_T = 200000;
    double sigma = 0.5;
    int write_chain = 0; /* 0/1 */
    double x0[2] = {0.0, 0.0};

    /* argv: TOTAL_T sigma write_chain */
    if (argc > 1) {
        long long tmpT = atoll(argv[1]);
        if (tmpT > 0) TOTAL_T = tmpT;
    }
    if (argc > 2) {
        double tmpS = atof(argv[2]);
        if (tmpS > 0.0) sigma = tmpS;
    }
    if (argc > 3) {
        write_chain = (atoi(argv[3]) != 0);
    }

    int nthreads;
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }

    int T_per_chain = (int)(TOTAL_T / nthreads);
    if (T_per_chain < 2) T_per_chain = 2;

    double *acc_rates = (double *)malloc((size_t)nthreads * sizeof(double));
    int    *accepted  = (int *)   malloc((size_t)nthreads * sizeof(int));
    double (*sum_x)[2]   = malloc((size_t)nthreads * sizeof *sum_x);
    double (*sumsq_x)[2] = malloc((size_t)nthreads * sizeof *sumsq_x);

    /* буфер цепочек (по потоку) — только если write_chain=1 */
    double **chains = NULL;
    if (write_chain) {
        chains = (double **)calloc((size_t)nthreads, sizeof(double *));
        if (!chains) {
            fprintf(stderr, "Allocation error (chains)\n");
            return 1;
        }
        for (int i = 0; i < nthreads; ++i) {
            chains[i] = (double *)malloc((size_t)2 * (size_t)T_per_chain * sizeof(double));
            if (!chains[i]) {
                fprintf(stderr, "Allocation error (chain buffer thread %d)\n", i);
                return 1;
            }
        }
    }

    if (!acc_rates || !accepted || !sum_x || !sumsq_x) {
        fprintf(stderr, "Allocation error\n");
        return 1;
    }

    /* ---------- измеряем только вычисления ---------- */
    double t0 = omp_get_wtime();

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int rng_state = 123456789u + 17u * (unsigned int)tid;

        int    loc_accepted = 0;
        double loc_sum_x[2] = {0.0, 0.0};
        double loc_sumsq_x[2] = {0.0, 0.0};

        double *chain_buf = (write_chain ? chains[tid] : NULL);

        double acc_rate = run_chain_compute(
            T_per_chain,
            sigma,
            x0,
            &rng_state,
            &loc_accepted,
            loc_sum_x,
            loc_sumsq_x,
            chain_buf
        );

        acc_rates[tid]  = acc_rate;
        accepted[tid]   = loc_accepted;
        sum_x[tid][0]   = loc_sum_x[0];
        sum_x[tid][1]   = loc_sum_x[1];
        sumsq_x[tid][0] = loc_sumsq_x[0];
        sumsq_x[tid][1] = loc_sumsq_x[1];
    }

    double t1 = omp_get_wtime();
    double elapsed_compute = t1 - t0;

    /* ---------- после измерения времени: запись файлов (если нужно) ---------- */
    if (write_chain) {
        /* последовательно, чтобы не убивать FS множеством одновременных fopen/fprintf */
        for (int tid = 0; tid < nthreads; ++tid) {
            write_chain_file(tid, T_per_chain, chains[tid]);
        }
    }

    /* ---------- глобальные статистики ---------- */
    long long N = (long long)(T_per_chain - 1) * (long long)nthreads;
    if (N <= 0) N = 1;

    long long total_accepted = 0;
    double global_sum_x0 = 0.0, global_sum_x1 = 0.0;
    double global_sumsq_x0 = 0.0, global_sumsq_x1 = 0.0;
    double sum_rates = 0.0;

    for (int i = 0; i < nthreads; ++i) {
        total_accepted  += accepted[i];
        global_sum_x0   += sum_x[i][0];
        global_sum_x1   += sum_x[i][1];
        global_sumsq_x0 += sumsq_x[i][0];
        global_sumsq_x1 += sumsq_x[i][1];
        sum_rates       += acc_rates[i];
    }

    double mean0 = global_sum_x0 / (double)N;
    double mean1 = global_sum_x1 / (double)N;
    double var0  = global_sumsq_x0 / (double)N - mean0 * mean0;
    double var1  = global_sumsq_x1 / (double)N - mean1 * mean1;

    double acc_rate_global = (double)total_accepted / (double)N;
    double avg_rate = sum_rates / (double)nthreads;

#ifdef VERBOSE
    printf("Metropolis + OpenMP\n");
    printf("Max threads available: %d\n", omp_get_max_threads());
    printf("TOTAL_T (sum over all chains) = %lld\n", TOTAL_T);
    printf("Proposal sigma = %f\n\n", sigma);

    printf("Running with %d OpenMP threads\n", nthreads);
    printf("T_per_chain = %d\n\n", T_per_chain);

    printf("Acceptance rates per thread:\n");
    for (int i = 0; i < nthreads; ++i) {
        printf("  thread %d: %.4f\n", i, acc_rates[i]);
    }
    printf("Average acceptance rate (by threads): %.4f\n\n", avg_rate);

    printf("Global stats over all threads:\n");
    printf("  Acceptance rate (per step): %.6f\n", acc_rate_global);
    printf("  Mean:     [%.6f, %.6f]\n", mean0, mean1);
    printf("  Variance: [%.6f, %.6f]\n\n", var0, var1);

    printf("Elapsed time (OpenMP, compute-only): %.6f seconds\n", elapsed_compute);
#else
    long long TOTAL_USED = (long long)T_per_chain * (long long)nthreads;
    printf("OMP=%d TOTAL_T=%lld elapsed=%.6f\n", nthreads, TOTAL_USED, elapsed_compute);
#endif

    if (chains) {
        for (int i = 0; i < nthreads; ++i) free(chains[i]);
        free(chains);
    }

    free(acc_rates);
    free(accepted);
    free(sum_x);
    free(sumsq_x);

    return 0;
}
