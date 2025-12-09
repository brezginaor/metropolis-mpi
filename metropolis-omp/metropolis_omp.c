#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------- RNG на потоке: rand_r + Бокс–Мюллер ---------- */

/* равномерное [0,1), state — локальное состояние для потока */
static inline double rng_uniform01(unsigned int *state)
{
    return (double)rand_r(state) / ((double)RAND_MAX + 1.0);
}

/* нормальное N(0,1) через Бокс–Мюллер */
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
/* x = (x0, x1), возвращаем log π(x) с точностью до константы */

double log_pi(const double x[2])
{
    /* log π(x) = const - 0.5 * (x0^2 + x1^2) — константу можно отбросить */
    return -0.5 * (x[0] * x[0] + x[1] * x[1]);
}

/* предложение: y = x + N(0, sigma^2 I) */
void q_sample(const double x[2], double y[2], double sigma, unsigned int *rng_state)
{
    y[0] = x[0] + sigma * rng_normal01(rng_state);
    y[1] = x[1] + sigma * rng_normal01(rng_state);
}

/* один шаг Метрополиса */
void metropolis_step(double x[2],
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
        /* принимаем безусловно */
        x[0]   = y[0];
        x[1]   = y[1];
        *logp  = logp_new;
        *accepted = 1;
    } else {
        double u = rng_uniform01(rng_state);
        if (u < exp(log_r)) {
            x[0]   = y[0];
            x[1]   = y[1];
            *logp  = logp_new;
            *accepted = 1;
        } else {
            *accepted = 0;
        }
    }
}

/* одна цепочка на одном потоке:
   - старт из x0
   - длина T
   - шаг sigma
   - свой генератор rng_state
   - выдаём:
       * коэффициент принятия (return)
       * accepted_total_out — количество принятых шагов
       * sum_x_out, sumsq_x_out — суммы для mean/variance
   - пишем координаты в файл chain_omp_thread<tid>.dat
*/
double run_chain(int thread_id,
                 int T,
                 double sigma,
                 const double x0[2],
                 unsigned int *rng_state,
                 int *accepted_total_out,
                 double sum_x_out[2],
                 double sumsq_x_out[2])
{
    double x[2]   = {x0[0], x0[1]};
    double logp   = log_pi(x);

    int accepted_count = 0;
    int acc = 0;

    double sum_x[2]   = {0.0, 0.0};
    double sumsq_x[2] = {0.0, 0.0};

    char filename[64];
    snprintf(filename, sizeof(filename), "chain_omp_thread%d.dat", thread_id);
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Thread %d: cannot open output file %s\n",
                thread_id, filename);
        return -1.0;
    }

    fprintf(f, "# t x0 x1\n");
    fprintf(f, "0 %.10f %.10f\n", x[0], x[1]);

    for (int t = 1; t < T; ++t) {
        metropolis_step(x, &logp, sigma, &acc, rng_state);
        accepted_count += acc;

        sum_x[0]   += x[0];
        sum_x[1]   += x[1];
        sumsq_x[0] += x[0] * x[0];
        sumsq_x[1] += x[1] * x[1];

        fprintf(f, "%d %.10f %.10f\n", t, x[0], x[1]);
    }

    fclose(f);

    if (accepted_total_out) {
        *accepted_total_out = accepted_count;
    }
    if (sum_x_out) {
        sum_x_out[0] = sum_x[0];
        sum_x_out[1] = sum_x[1];
    }
    if (sumsq_x_out) {
        sumsq_x_out[0] = sumsq_x[0];
        sumsq_x_out[1] = sumsq_x[1];
    }

    double acc_rate = (double)accepted_count / (double)(T - 1);
    return acc_rate;
}

/* -------------------------- main с OpenMP -------------------------- */

int main(int argc, char *argv[])
{
    /* TOTAL_T — суммарная длина по ВСЕМ цепям (как в MPI-варианте) */
    long long TOTAL_T = 200000;  /* по умолчанию */
    double sigma = 0.5;          /* шаг предложения */
    double x0[2] = {0.0, 0.0};

    if (argc > 1) {
        long long tmpT = atoll(argv[1]);
        if (tmpT > 0) TOTAL_T = tmpT;
    }
    if (argc > 2) {
        double tmpS = atof(argv[2]);
        if (tmpS > 0.0) sigma = tmpS;
    }

    int max_threads = omp_get_max_threads();

    printf("Metropolis + OpenMP\n");
    printf("Max threads available: %d\n", max_threads);
    printf("TOTAL_T (sum over all chains) = %lld\n", TOTAL_T);
    printf("Proposal sigma = %f\n\n", sigma);

    int nthreads;
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }

    printf("Running with %d OpenMP threads\n", nthreads);

    // шагов на одну цепочку (как T_local = TOTAL_T / NP) 
    int T_per_chain = (int)(TOTAL_T / nthreads);
    if (T_per_chain < 2) T_per_chain = 2;  

    printf("T_per_chain = %d\n\n", T_per_chain);

    //массивы для сбора статистик от каждого потока 
    double *acc_rates = (double *)malloc(nthreads * sizeof(double));
    int    *accepted  = (int *)   malloc(nthreads * sizeof(int));
    double (*sum_x)[2]   = malloc(nthreads * sizeof *sum_x);
    double (*sumsq_x)[2] = malloc(nthreads * sizeof *sumsq_x);

    if (!acc_rates || !accepted || !sum_x || !sumsq_x) {
        fprintf(stderr, "Allocation error\n");
        return 1;
    }

    double t0 = omp_get_wtime();

// по умолчанию все переменные приватные внутри
#pragma omp parallel
    {
        //возвр номер потока 
        int tid = omp_get_thread_num();

        //свой seed у каждого потока 
        unsigned int rng_state = 123456789u + 17u * (unsigned int)tid;

        //Локальные переменные статистики
        int    loc_accepted;
        double loc_sum_x[2];
        double loc_sumsq_x[2];

        double acc_rate = run_chain(
            tid,
            T_per_chain,
            sigma,
            x0,
            &rng_state,
            &loc_accepted,
            loc_sum_x,
            loc_sumsq_x
        );

        // запись результатов    этого потока в глобальные массивы 
        acc_rates[tid]   = acc_rate;
        accepted[tid]    = loc_accepted;
        sum_x[tid][0]    = loc_sum_x[0];
        sum_x[tid][1]    = loc_sum_x[1];
        sumsq_x[tid][0]  = loc_sumsq_x[0];
        sumsq_x[tid][1]  = loc_sumsq_x[1];
    }

    double t1 = omp_get_wtime();

    /* ---------- печать per-thread acceptance ---------- */
    printf("Acceptance rates per thread:\n");
    double sum_rates = 0.0;
    for (int i = 0; i < nthreads; ++i) {
        printf("  thread %d: %.4f\n", i, acc_rates[i]);
        sum_rates += acc_rates[i];
    }
    printf("Average acceptance rate (by threads): %.4f\n",
           sum_rates / (double)nthreads);

    /* ---------- глобальные статистики ---------- */

    long long N = (long long)(T_per_chain - 1) * (long long)nthreads;

    long long total_accepted = 0;
    double global_sum_x0 = 0.0, global_sum_x1 = 0.0;
    double global_sumsq_x0 = 0.0, global_sumsq_x1 = 0.0;

    for (int i = 0; i < nthreads; ++i) {
        total_accepted   += accepted[i];
        global_sum_x0    += sum_x[i][0];
        global_sum_x1    += sum_x[i][1];
        global_sumsq_x0  += sumsq_x[i][0];
        global_sumsq_x1  += sumsq_x[i][1];
    }

    double mean0 = global_sum_x0 / (double)N;
    double mean1 = global_sum_x1 / (double)N;

    double var0 = global_sumsq_x0 / (double)N - mean0 * mean0;
    double var1 = global_sumsq_x1 / (double)N - mean1 * mean1;

    double acc_rate_global = (double)total_accepted / (double)N;

    printf("\nGlobal stats over all threads:\n");
    printf("  Acceptance rate (per step): %.6f\n", acc_rate_global);
    printf("  Mean:     [%.6f, %.6f]\n", mean0, mean1);
    printf("  Variance: [%.6f, %.6f]\n", var0, var1);

    printf("\nElapsed time (OpenMP): %.6f seconds\n", t1 - t0);

    free(acc_rates);
    free(accepted);
    free(sum_x);
    free(sumsq_x);

    return 0;
}
