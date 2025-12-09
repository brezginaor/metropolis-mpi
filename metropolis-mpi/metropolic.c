#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "sprng.h"

#define SEED 123456789

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------- Глобальный генератор SPRNG для каждого процесса ---------- */

static int *rng_stream = NULL;

void init_rng_stream(int rank, int size)
{
    /* int *init_rng(int gennum, int total_gen, int seed, int param); */
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

/* равномерное [0,1) */
double rng_uniform01(void)
{
    return sprng(rng_stream);
}

/* нормальное N(0,1) через Бокс–Мюллер */
double rng_normal01(void)
{
    double u1 = rng_uniform01();
    double u2 = rng_uniform01();

    if (u1 < 1e-12) u1 = 1e-12;

    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;

    return r * cos(theta);
}

/* лог-плотность π(x) ~ N(0, I) в 2D, x = (x0, x1) */
double log_pi(const double x[2])
{
    return -0.5 * (x[0] * x[0] + x[1] * x[1]);
}

/* предложение: y = x + N(0, sigma^2 I) */
void q_sample(const double x[2], double y[2], double sigma)
{
    y[0] = x[0] + sigma * rng_normal01();
    y[1] = x[1] + sigma * rng_normal01();
}

/* один шаг Метрополиса */
void metropolis_step(double x[2],
                     double *logp,
                     double sigma,
                     int *accepted)
{
    double y[2];
    q_sample(x, y, sigma);

    double logp_new = log_pi(y);   /* <-- здесь была опечатка "ouble" */
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

/* одна цепочка на одном ранге, плюс локальные суммы для статистик */
double run_chain(const double x0[2], double sigma, int T, int rank,
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

    char filename[64];
    snprintf(filename, sizeof(filename), "chain_rank%d.dat", rank);
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Rank %d: cannot open output file %s\n", rank, filename);
        return -1.0;
    }

    fprintf(f, "# t x0 x1\n");
    fprintf(f, "0 %.10f %.10f\n", x[0], x[1]);

    for (int t = 1; t < T; ++t) {
        metropolis_step(x, &logp, sigma, &acc);
        accepted_count += acc;

        /* копим суммы для mean/variance */
        sum_x[0]   += x[0];
        sum_x[1]   += x[1];
        sumsq_x[0] += x[0] * x[0];
        sumsq_x[1] += x[1] * x[1];

        fprintf(f, "%d %.10f %.10f\n", t, x[0], x[1]);
    }

    fclose(f);

    if (accepted_total_out != NULL) {
        *accepted_total_out = accepted_count;
    }
    if (sum_x_out != NULL) {
        sum_x_out[0] = sum_x[0];
        sum_x_out[1] = sum_x[1];
    }
    if (sumsq_x_out != NULL) {
        sumsq_x_out[0] = sumsq_x[0];
        sumsq_x_out[1] = sumsq_x[1];
    }

    double acc_rate = (double)accepted_count / (double)(T - 1);
    return acc_rate;
}

/* ---------- main с MPI, Gather и ручным Send/Recv ---------- */

int main(int argc, char *argv[])
{
    int rank, size;

    /* TOTAL_T — общее число шагов по всем процессам вместе */
    long long TOTAL_T = 1000;    /* по умолчанию */
    double sigma = 0.5;          /* шаг предложения */
    double x0[2] = {0.0, 0.0};   /* стартовая точка */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t_start = MPI_Wtime();

    /* аргументы командной строки:
       argv[1] — TOTAL_T (общее число шагов по всем ранкам)
       argv[2] — sigma
       пример: mpirun -np 4 ./metropolic 200000 0.5
    */
    if (argc > 1) {
        long long tmpT = atoll(argv[1]);
        if (tmpT > 0) TOTAL_T = tmpT;
    }
    if (argc > 2) {
        double tmpS = atof(argv[2]);
        if (tmpS > 0.0) sigma = tmpS;
    }

    /* делим TOTAL_T по ранкам как можно равномернее:
       base = floor(TOTAL_T / size)
       первые rem рангов получают +1 шаг
    */
    long long base = TOTAL_T / size;
    int rem        = (int)(TOTAL_T % size);

    int T_local = (int)base + (rank < rem ? 1 : 0);
    int N_local = (T_local > 0 ? T_local - 1 : 0);  /* сколько точек реально идёт в статистику */

    if (rank == 0) {
        printf("Metropolis + MPI + SPRNG\n");
        printf("Processes: %d\n", size);
        printf("TOTAL_T (sum over ranks): %lld\n", TOTAL_T);
        printf("Local T on rank 0: %d\n", T_local);
        printf("Proposal sigma: %f\n", sigma);
    }

    /* инициализация SPRNG для каждого процесса */
    init_rng_stream(rank, size);

    /* локальные статистики для этого ранга */
    int    local_accepted   = 0;
    double local_sum_x[2]   = {0.0, 0.0};
    double local_sumsq_x[2] = {0.0, 0.0};

    /* запускаем локальную цепочку длиной T_local */
    double local_acc_rate = run_chain(x0,
                                      sigma,
                                      T_local,
                                      rank,
                                      &local_accepted,
                                      local_sum_x,
                                      local_sumsq_x);

    /* ---------- Собираем acceptance rate по рангам (через MPI_Gather) ---------- */
    double *all_rates = NULL;
    if (rank == 0) {
        all_rates = (double *)malloc(size * sizeof(double));
    }

    MPI_Gather(&local_acc_rate, 1, MPI_DOUBLE,
               all_rates,        1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nAcceptance rates per rank:\n");
        double sum_rates = 0.0;
        for (int i = 0; i < size; ++i) {
            printf("  rank %d: %.4f\n", i, all_rates[i]);
            sum_rates += all_rates[i];
        }
        printf("Average acceptance rate (by ranks): %.4f\n",
               sum_rates / (double)size);
        free(all_rates);
    }

    /* ---------- Глобальные статистики через ручной Send/Recv ---------- */

    if (rank == 0) {
        /* инициализируем глобальные суммы своими локальными значениями */
        int    global_accepted   = local_accepted;
        double global_sum_x[2]   = { local_sum_x[0],   local_sum_x[1]   };
        double global_sumsq_x[2] = { local_sumsq_x[0], local_sumsq_x[1] };
        long long global_N       = (long long)N_local;

        /* принимаем данные от рангов 1..size-1 */
        for (int src = 1; src < size; ++src) {
            int    recv_accepted;
            double recv_sum_x[2];
            double recv_sumsq_x[2];
            int    recv_N;

            /* сначала число принятых шагов */
            MPI_Recv(&recv_accepted, 1, MPI_INT,
                     src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* затем сумма координат */
            MPI_Recv(recv_sum_x, 2, MPI_DOUBLE,
                     src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* затем сумма квадратов координат */
            MPI_Recv(recv_sumsq_x, 2, MPI_DOUBLE,
                     src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /* а теперь N_local с этого ранга */
            MPI_Recv(&recv_N, 1, MPI_INT,
                     src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            global_accepted   += recv_accepted;
            global_sum_x[0]   += recv_sum_x[0];
            global_sum_x[1]   += recv_sum_x[1];
            global_sumsq_x[0] += recv_sumsq_x[0];
            global_sumsq_x[1] += recv_sumsq_x[1];
            global_N          += (long long)recv_N;
        }

        /* считаем глобальные статистики по всем сэмплам */
        double N_as_double = (double)global_N;

        double mean0 = global_sum_x[0] / N_as_double;
        double mean1 = global_sum_x[1] / N_as_double;

        double var0 = global_sumsq_x[0] / N_as_double - mean0 * mean0;
        double var1 = global_sumsq_x[1] / N_as_double - mean1 * mean1;

        double acc_rate_global = (double)global_accepted / N_as_double;

        printf("\nGlobal stats over all ranks (via Send/Recv):\n");
        printf("  Acceptance rate (per step): %.6f\n", acc_rate_global);
        printf("  Mean:     [%.6f, %.6f]\n", mean0, mean1);
        printf("  Variance: [%.6f, %.6f]\n", var0, var1);
    } else {
        /* не-нолевые ранги отсылают свои локальные суммы рангу 0 */

        /* число принятых шагов */
        MPI_Send(&local_accepted, 1, MPI_INT,
                 0, 0, MPI_COMM_WORLD);

        /* суммы координат */
        MPI_Send(local_sum_x, 2, MPI_DOUBLE,
                 0, 1, MPI_COMM_WORLD);

        /* суммы квадратов координат */
        MPI_Send(local_sumsq_x, 2, MPI_DOUBLE,
                 0, 2, MPI_COMM_WORLD);

        /* количество использованных точек N_local = T_local - 1 */
        MPI_Send(&N_local, 1, MPI_INT,
                 0, 3, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    if (rank == 0) {
        printf("\nElapsed time: %.6f seconds\n", t_end - t_start);
    }

    free_rng_stream();
    MPI_Finalize();
    return 0;
}

