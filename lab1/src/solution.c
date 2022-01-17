#include "solution.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>


double *gauss(double **A_matrix, const double *y, size_t nodes_num, size_t threads_num) {
    double *x_sol = (double *) calloc(sizeof(double), nodes_num);
    double *b_vec = (double *) calloc(sizeof(double), nodes_num);
    double **A_mtrx_tmp = (double **) calloc(sizeof(double *), nodes_num);

    for (int i = 0; i < nodes_num; ++i) {
        A_mtrx_tmp[i] = (double *) calloc(sizeof(double), nodes_num);
    }


    for (int i = 0; i < nodes_num; i++) {
        for (int j = 0; j < nodes_num; j++) {
            A_mtrx_tmp[i][j] = A_matrix[i][j];
        }
    }

    for (int i = 0; i < nodes_num; i++) {
        b_vec[i] = y[i];
    }

    for (int k = 1; k < nodes_num; ++k) { // прямой ход
        int j;
#pragma omp parallel for shared(A_mtrx_tmp, b_vec, k, nodes_num) default(none) num_threads(threads_num)
        for (j = k + 1; j < nodes_num; ++j) {
            double pivot = A_mtrx_tmp[j][k] / A_mtrx_tmp[k][k];
            printf("check pivot: %lf\n", A_mtrx_tmp[k][k]);

            for (int i = k; i < nodes_num; i++) {
                A_mtrx_tmp[j][i] -= pivot * A_mtrx_tmp[k][i];
            }

            b_vec[j] = b_vec[j] - pivot * b_vec[k];
        }
    }

    for (int k = nodes_num - 1; k >= 1; k--) { // обратный ход
        double sum = 0;
        int j;
#pragma omp parallel for shared(A_mtrx_tmp, k, nodes_num, x_sol) default(none) num_threads(threads_num) reduction(+:sum)
        for (j = k + 1; j < nodes_num; j++) {
            sum += A_mtrx_tmp[k][j] * x_sol[j];
        }

        x_sol[k] = (b_vec[k] - sum) / A_mtrx_tmp[k][k];
    }

    free(b_vec);

    for (int i = 0; i < nodes_num; i++) {
        free(A_mtrx_tmp[i]);
    }

    free(A_mtrx_tmp);

    return x_sol;
}

double *solution(double *y, double **A, struct solution_params *params, size_t threads_num, double *performance_stats) {
    size_t y_nodes = (int) (params->height / params->step + 1);
    size_t x_nodes = (int) (params->width / params->step + 1);

    size_t nodes_amount = (int) x_nodes * y_nodes;

    double *x = (double *) calloc(sizeof(double), nodes_amount);

    double t1 = omp_get_wtime();
    for (int k = 0; k < params->time_end; ++k) {
        for (int i = 0; i < y_nodes; ++i) {
            for (int j = 0; j < x_nodes; ++j) {
                if (j > 0 && i > 0 && i < y_nodes - 1 && i > (j - params->top_border / params->step) && j < (x_nodes - 1)) {
                    y[i * x_nodes + j] = x[i * x_nodes + j];
                }
            }
        }
        free(x);

        x = gauss(A, y, nodes_amount, threads_num);
    }

    double t2 = omp_get_wtime();
    *performance_stats = t2 - t1;
    printf("[Performance] threads: '%llu': %f secs\n", threads_num, *performance_stats);
    return x;
}

void initialise(double **vector, double ***matrix, struct solution_params *params) {
    size_t y_nodes = (int) (params->height / params->step + 1);
    size_t x_nodes = (int) (params->width / params->step + 1);

    size_t nodes_num = (int) x_nodes * y_nodes;

    for (int i = 0; i < y_nodes; i++) {
        for (int j = 0; j < x_nodes; j++) {
            if (i == 0 && j <= params->top_border / params->step) { // Верхняя граница - ГУ 1-го рода
                (*vector)[i * x_nodes + j] = 50; // Само значение в векторе температур
                (*matrix)[i * x_nodes + j][i * x_nodes + j] = 1; // Указание этого уравнения в СЛАУ
            } else if (j == 0) { // Левая граница - ГУ 1-го рода
                (*vector)[i * x_nodes + j] = 100;
                (*matrix)[i * x_nodes + j][i * x_nodes + j] = 1;
            } else if (i == y_nodes - 1) { // Нижняя граница - ГУ 3-го рода (-1 т.к. индексация с нуля)
                (*matrix)[i * x_nodes + j][i * x_nodes + j] = -1 / params->step - 1;
                (*matrix)[i * x_nodes + j][(i - 1) * x_nodes + j] = 1 / params->step;
                (*vector)[i * x_nodes + j] = params->temp_init; // начальное условие нестационарной задачи
            } else if (j >= params->top_border / params->step && i == (j - params->top_border / params->step)) { // Правая скошенная граница - ГУ 2-го рода
                double cos_45 = sqrt(2) / 2;
                (*matrix)[i * x_nodes + j][i * x_nodes + j] = 1 / (sqrt(2) * params->step);
                (*matrix)[i * x_nodes + j][(i + 1) * x_nodes + j - 1] = -1 / (sqrt(2) * params->step);
                (*vector)[i * x_nodes + j] = 20;
            } else if (i * params->step <= (params->height - params->right_border) && j >= params->top_border / params->step &&
                       i <= (j - params->top_border / params->step)) { // Пустая область
                (*matrix)[i * x_nodes + j][i * x_nodes + j] = 1;
                (*vector)[i * x_nodes + j] = 0;
            } else if (j == x_nodes - 1) { // Правая граница - ГУ 2-го рода
                (*matrix)[i * x_nodes + j][i * x_nodes + j] = 1 / params->step;
                (*matrix)[i * x_nodes + j][i * x_nodes + j - 1] = -1 / params->step;
                (*vector)[i * x_nodes + j] = 20;
            } else if (j < x_nodes - 1) { // Остальные (внутренние) узлы
                (*matrix)[i * x_nodes + j][i * x_nodes + j] = 1 + 2 / (params->step * params->step) + 2 / (params->step * params->step);
                (*matrix)[i * x_nodes + j][i * x_nodes + j + 1] = -1 / (params->step * params->step);
                (*matrix)[i * x_nodes + j][i * x_nodes + j - 1] = -1 / (params->step * params->step);
                (*matrix)[i * x_nodes + j][(i + 1) * x_nodes + j] = -1 / (params->step * params->step);
                (*matrix)[i * x_nodes + j][(i - 1) * x_nodes + j] = -1 / (params->step * params->step);
            }
        }
    }

    printf("CHECK INIT: \n");
    for (int row = 0; row < nodes_num; ++row) {
        printf("matrix value: %lf\n", (*matrix)[row][row]);
    }
}