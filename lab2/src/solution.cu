#include "solution.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ctime>


__global__ void gauss(double *A_matrix, double *y, double *x, size_t nodes_num, double* reduction) {
    for (int k = 1; k < nodes_num; ++k) { // прямой ход
        for (int j = k + 1 + threadIdx.x; j < nodes_num; j += blockDim.x) {
            double pivot = A_matrix[j * nodes_num + k] / A_matrix[k * nodes_num + k];
            for (int i = k; i < nodes_num; i++) {
                A_matrix[j * nodes_num + i] -= pivot * A_matrix[k * nodes_num + i];
            }

            y[j] = y[j] - pivot * y[k];
        }
    }

    for (int k = nodes_num - 1; k >= 1; k--) { // обратный ход
        __shared__ double total_sum;
        double sum = 0;
        for (int j = k + 1 + threadIdx.x; j < nodes_num; j += blockDim.x) {
            sum += A_matrix[k * nodes_num + j] * x[j];
        }

        reduction[threadIdx.x] = sum;
        __syncthreads();
        for (int size = blockDim.x / 2; size > 0; size /= 2) { //uniform
            if (threadIdx.x < size) {
                reduction[threadIdx.x] += reduction[threadIdx.x + size];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            total_sum = reduction[0];
        }
        __syncthreads();

        x[k] = (y[k] - total_sum) / A_matrix[k * nodes_num + k];
    }
}

double *solution(double *y, double *A, struct solution_params *params, size_t threads_num, double *performance_stats) {
    size_t y_nodes = (int) (params->height / params->step + 1);
    size_t x_nodes = (int) (params->width / params->step + 1);

    size_t nodes_amount = (int) x_nodes * y_nodes;

    double *x = (double *) calloc(sizeof(double), nodes_amount);
    double *x_device = nullptr;
    double *y_device = nullptr;
    double *A_device = nullptr;

    // Массив для суммирования в обратном ходе Гаусса (в случае параллелизма необходима редукция)
    double *reduction_arr = nullptr;

    HANDLE_ERROR(cudaMalloc((void **) &x_device, nodes_amount * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &y_device, nodes_amount * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &reduction_arr, threads_num * sizeof(double)));
    HANDLE_ERROR(cudaMalloc((void **) &A_device, nodes_amount * nodes_amount * sizeof(double)));

    clock_t t = clock();
    for (int k = 0; k < params->time_end; ++k) {
        for (int i = 0; i < y_nodes; ++i) {
            for (int j = 0; j < x_nodes; ++j) {
                if (j > 0 && i > 0 && i < y_nodes - 1 && i > (j - params->top_border / params->step) && j < (x_nodes - 1)) {
                    y[i * x_nodes + j] = x[i * x_nodes + j];
                }
            }
        }

        HANDLE_ERROR(cudaMemcpy((void *) y_device, (void *) y, nodes_amount * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy((void *) x_device, (void *) x, nodes_amount * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy((void *) A_device, (void *) A, nodes_amount * nodes_amount * sizeof(double), cudaMemcpyHostToDevice));

        gauss<<<1, threads_num>>>(A_device, y_device, x_device, nodes_amount, reduction_arr);

//        HANDLE_ERROR(cudaDeviceSynchronize());
        HANDLE_ERROR(cudaMemcpy((void *) x, (void *) x_device, nodes_amount * sizeof(double), cudaMemcpyDeviceToHost));
    }

    double total_time = (double) (clock() - t) / CLOCKS_PER_SEC;
    *performance_stats = total_time;
    printf("[Performance] threads: '%llu': %f secs\n", threads_num, *performance_stats);
    return x;
}

void initialise(double **vector, double **matrix, struct solution_params *params) {
    size_t y_nodes = (int) (params->height / params->step + 1);
    size_t x_nodes = (int) (params->width / params->step + 1);

    size_t nodes_amount = (int) x_nodes * y_nodes;

    for (int i = 0; i < y_nodes; i++) {
        for (int j = 0; j < x_nodes; j++) {
            if (i == 0 && j <= params->top_border / params->step) { // Верхняя граница - ГУ 1-го рода
                (*vector)[i * x_nodes + j] = 50; // Само значение в векторе температур
                (*matrix)[(i * x_nodes + j) * nodes_amount + (i * x_nodes + j)] = 1; // Указание этого уравнения в СЛАУ
            } else if (j == 0) { // Левая граница - ГУ 1-го рода
                (*vector)[i * x_nodes + j] = 100;
                (*matrix)[(i * x_nodes + j) * nodes_amount + (i * x_nodes + j)] = 1;
            } else if (i == y_nodes - 1) { // Нижняя граница - ГУ 3-го рода (-1 т.к. индексация с нуля)
                (*matrix)[(i * x_nodes + j) * nodes_amount + (i * x_nodes + j)] = -1 / params->step - 1;
                (*matrix)[(i * x_nodes + j) * nodes_amount + ((i - 1) * x_nodes + j)] = 1 / params->step;
                (*vector)[i * x_nodes + j] = params->temp_init; // начальное условие нестационарной задачи
            } else if (j >= params->top_border / params->step && i == (j - params->top_border / params->step)) { // Правая скошенная граница - ГУ 2-го рода
                double cos_45 = sqrt(2) / 2;
                (*matrix)[(i * x_nodes + j) * nodes_amount + (i * x_nodes + j)] = 1 / (sqrt(2) * params->step);
                (*matrix)[(i * x_nodes + j) * nodes_amount + ((i + 1) * x_nodes + j - 1)] = -1 / (sqrt(2) * params->step);
                (*vector)[i * x_nodes + j] = 20;
            } else if (i * params->step <= (params->height - params->right_border) && j >= params->top_border / params->step &&
                       i <= (j - params->top_border / params->step)) { // Пустая область
                (*matrix)[(i * x_nodes + j) * nodes_amount + (i * x_nodes + j)] = 1;
                (*vector)[i * x_nodes + j] = 0;
            } else if (j == x_nodes - 1) { // Правая граница - ГУ 2-го рода
                (*matrix)[(i * x_nodes + j) * nodes_amount + (i * x_nodes + j)] = 1 / params->step;
                (*matrix)[(i * x_nodes + j) * nodes_amount + (i * x_nodes + j - 1)] = -1 / params->step;
                (*vector)[i * x_nodes + j] = 20;
            } else if (j < x_nodes - 1) { // Остальные (внутренние) узлы
                (*matrix)[(i * x_nodes + j) * nodes_amount + (i * x_nodes + j)] = 1 + 2 / (params->step * params->step) + 2 / (params->step * params->step);
                (*matrix)[(i * x_nodes + j) * nodes_amount + (i * x_nodes + j + 1)] = -1 / (params->step * params->step);
                (*matrix)[(i * x_nodes + j) * nodes_amount + (i * x_nodes + j - 1)] = -1 / (params->step * params->step);
                (*matrix)[(i * x_nodes + j) * nodes_amount + ((i + 1) * x_nodes + j)] = -1 / (params->step * params->step);
                (*matrix)[(i * x_nodes + j) * nodes_amount + ((i - 1) * x_nodes + j)] = -1 / (params->step * params->step);
            }
        }
    }
//    printf("CHECK INIT: \n");
//    for (int row = 0; row < nodes_amount; ++row) {
//        printf("matrix value: %lf\n", (*matrix)[row * nodes_amount + row]);
//    }
}