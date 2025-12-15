#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include "sprng.h"

#define SEED 123456789

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    int tid;
    int nthreads;

    int T_local;
    double sigma;
    int write_chain;

    // результаты
    int accepted;
    double sum_x0, sum_x1;
    double sumsq_x0, sumsq_x1;
    int samples_used;
    double acc_rate;
} thread_ctx_t;

/* ---------- RNG (SPRNG) per-thread ---------- */

static inline double rng_uniform01(int *stream) {
    return sprng(stream); // [0,1)
}

static inline double rng_normal01(int *stream) {
    // Box–Muller
    double u1 = rng_uniform01(stream);
    double u2 = rng_uniform01(stream);
    if (u1 < 1e-12) u1 = 1e-12;
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    return r * cos(theta);
}

/* log π(x) ~ N(0,I) in 2D */
static inline double log_pi(double x0, double x1) {
    return -0.5 * (x0*x0 + x1*x1);
}

static inline void q_sample(double x0, double x1, double sigma, int *stream,
                            double *y0, double *y1) {
    *y0 = x0 + sigma * rng_normal01(stream);
    *y1 = x1 + sigma * rng_normal01(stream);
}

static inline void metropolis_step(double *x0, double *x1, double *logp,
                                   double sigma, int *stream, int *accepted) {
    double y0, y1;
    q_sample(*x0, *x1, sigma, stream, &y0, &y1);

    double logp_new = log_pi(y0, y1);
    double log_r = logp_new - (*logp);

    if (log_r >= 0.0) {
        *x0 = y0; *x1 = y1; *logp = logp_new; *accepted = 1;
        return;
    }

    double u = rng_uniform01(stream);
    if (u < exp(log_r)) {
        *x0 = y0; *x1 = y1; *logp = logp_new; *accepted = 1;
    } else {
        *accepted = 0;
    }
}

/* ---------- Thread body ---------- */

static void *thread_main(void *arg) {
    thread_ctx_t *ctx = (thread_ctx_t *)arg;

    // SPRNG stream per thread:
    // init_rng(gennum, total_gen, seed, param)
    int *stream = init_rng(ctx->tid, ctx->nthreads, SEED, 0);
    if (!stream) {
        fprintf(stderr, "Thread %d: init_rng failed\n", ctx->tid);
        pthread_exit((void*)1);
    }

    double x0 = 0.0, x1 = 0.0;
    double logp = log_pi(x0, x1);

    int accepted_total = 0;
    double sum_x0 = 0.0, sum_x1 = 0.0;
    double sumsq_x0 = 0.0, sumsq_x1 = 0.0;

    FILE *f = NULL;
    if (ctx->write_chain) {
        char name[128];
        snprintf(name, sizeof(name), "chain_thread%d.dat", ctx->tid);
        f = fopen(name, "w");
        if (!f) {
            fprintf(stderr, "Thread %d: cannot open %s\n", ctx->tid, name);
            free_rng(stream);
            pthread_exit((void*)2);
        }
        fprintf(f, "# t x0 x1\n");
        fprintf(f, "0 %.10f %.10f\n", x0, x1);
    }

    int samples_used = (ctx->T_local > 1) ? (ctx->T_local - 1) : 0;

    for (int t = 1; t < ctx->T_local; ++t) {
        int acc = 0;
        metropolis_step(&x0, &x1, &logp, ctx->sigma, stream, &acc);
        accepted_total += acc;

        sum_x0 += x0; sum_x1 += x1;
        sumsq_x0 += x0*x0; sumsq_x1 += x1*x1;

        if (f) fprintf(f, "%d %.10f %.10f\n", t, x0, x1);
    }

    if (f) fclose(f);

    ctx->accepted = accepted_total;
    ctx->sum_x0 = sum_x0; ctx->sum_x1 = sum_x1;
    ctx->sumsq_x0 = sumsq_x0; ctx->sumsq_x1 = sumsq_x1;
    ctx->samples_used = samples_used;
    ctx->acc_rate = (samples_used > 0) ? ((double)accepted_total / (double)samples_used) : 0.0;

    free_rng(stream);
    pthread_exit(NULL);
}

/* ---------- Timing ---------- */

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    long long TOTAL_T = 200000;  // суммарно по всем потокам
    double sigma = 0.5;
    int nthreads = 1;
    int write_chain = 0; // 0/1
    int quiet = 0;       // 0=verbose, 1=одна строка (для свипов)

    // argv: TOTAL_T sigma nthreads write_chain quiet
    if (argc > 1) {
        long long t = atoll(argv[1]);
        if (t > 0) TOTAL_T = t;
    }
    if (argc > 2) {
        double s = atof(argv[2]);
        if (s > 0.0) sigma = s;
    }
    if (argc > 3) {
        int p = atoi(argv[3]);
        if (p > 0) nthreads = p;
    }
    if (argc > 4) write_chain = (atoi(argv[4]) != 0);
    if (argc > 5) quiet = (atoi(argv[5]) != 0);

    if (nthreads < 1) nthreads = 1;

    // распределяем TOTAL_T по потокам
    long long base = TOTAL_T / nthreads;
    long long rem  = TOTAL_T % nthreads;

    pthread_t *ths = (pthread_t *)calloc(nthreads, sizeof(pthread_t));
    thread_ctx_t *ctx = (thread_ctx_t *)calloc(nthreads, sizeof(thread_ctx_t));
    if (!ths || !ctx) {
        fprintf(stderr, "Allocation error\n");
        free(ths); free(ctx);
        return 1;
    }

    double t0 = wall_seconds();

    for (int i = 0; i < nthreads; ++i) {
        ctx[i].tid = i;
        ctx[i].nthreads = nthreads;
        ctx[i].sigma = sigma;
        ctx[i].write_chain = write_chain;

        int T_local = (int)(base + (i < rem ? 1 : 0));
        if (T_local < 2) T_local = 2;
        ctx[i].T_local = T_local;

        int rc = pthread_create(&ths[i], NULL, thread_main, &ctx[i]);
        if (rc != 0) {
            fprintf(stderr, "pthread_create failed for thread %d\n", i);
            return 2;
        }
    }

    for (int i = 0; i < nthreads; ++i) {
        pthread_join(ths[i], NULL);
    }

    double t1 = wall_seconds();
    double elapsed = t1 - t0;

    // суммируем статистики
    long long N = 0;
    long long total_accepted = 0;
    double sum_x0 = 0.0, sum_x1 = 0.0;
    double sumsq_x0 = 0.0, sumsq_x1 = 0.0;
    double sum_rates = 0.0;

    for (int i = 0; i < nthreads; ++i) {
        total_accepted += ctx[i].accepted;
        sum_x0 += ctx[i].sum_x0;
        sum_x1 += ctx[i].sum_x1;
        sumsq_x0 += ctx[i].sumsq_x0;
        sumsq_x1 += ctx[i].sumsq_x1;
        N += ctx[i].samples_used;
        sum_rates += ctx[i].acc_rate;
    }

    if (N <= 0) N = 1;

    double mean0 = sum_x0 / (double)N;
    double mean1 = sum_x1 / (double)N;
    double var0 = sumsq_x0 / (double)N - mean0 * mean0;
    double var1 = sumsq_x1 / (double)N - mean1 * mean1;
    double acc_global = (double)total_accepted / (double)N;
    double avg_rate = sum_rates / (double)nthreads;

    long long TOTAL_USED = 0;
    for (int i = 0; i < nthreads; ++i) TOTAL_USED += ctx[i].T_local;

    if (quiet) {
        printf("PTH=%d TOTAL_T=%lld elapsed=%.6f\n", nthreads, TOTAL_USED, elapsed);
    } else {
        printf("Metropolis + C + pthreads + SPRNG\n");
        printf("Threads: %d\n", nthreads);
        printf("TOTAL_T (requested): %lld\n", TOTAL_T);
        printf("TOTAL_T (used sum local): %lld\n", TOTAL_USED);
        printf("Proposal sigma: %.6f\n", sigma);
        printf("WRITE_CHAIN: %d\n\n", write_chain);

        printf("Acceptance rates per thread:\n");
        for (int i = 0; i < nthreads; ++i) {
            printf("  thread %d: %.4f\n", i, ctx[i].acc_rate);
        }
        printf("Average acceptance rate (by threads): %.4f\n\n", avg_rate);

        printf("Global stats over all threads:\n");
        printf("  Samples used: %lld\n", N);
        printf("  Acceptance rate (per step): %.6f\n", acc_global);
        printf("  Mean:     [%.6f, %.6f]\n", mean0, mean1);
        printf("  Variance: [%.6f, %.6f]\n\n", var0, var1);

        printf("Elapsed time (pthreads): %.6f seconds\n", elapsed);
    }

    free(ths);
    free(ctx);
    return 0;
}
