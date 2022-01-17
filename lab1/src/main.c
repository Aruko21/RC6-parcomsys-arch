#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "solution.h"
#include "utils.h"

#define PERFORMANCE_RESULTS_FILENAME "performance.dat"
#define SOLUTION_FILENAME "solution.dat"
#define THREADS_NUM 5

#define TIME_END 15
#define HEIGHT 6
#define WIDTH 8
#define RIGHT_BORDER 3
#define TOP_BORDER 5
#define STEP 1.5

#define ALPHA 1
#define TEMP_INIT 0


int main() {
    size_t y_nodes = (int) (HEIGHT / STEP + 1);
    size_t x_nodes = (int) (WIDTH / STEP + 1);

    size_t nodes_amount = (int) x_nodes * y_nodes;
    printf("Nodes amount: %llu\n", nodes_amount);

    double *temperatures = (double *) calloc(sizeof(double), nodes_amount);

    double **matrix = (double **) malloc(sizeof(double *) * nodes_amount);
    for (int i = 0; i < nodes_amount; ++i) {
        matrix[i] = (double *) calloc(sizeof(double), nodes_amount);
    }

    struct solution_params params = {
        .time_end = TIME_END,
        .step = STEP,
        .height = HEIGHT,
        .width = WIDTH,
        .top_border = TOP_BORDER,
        .right_border = RIGHT_BORDER,
        .temp_init = TEMP_INIT,
        .alpha = ALPHA
    };

    initialise(&temperatures, &matrix, &params);

    double *performance_stats = (double *) malloc(sizeof(double) * 1);
    size_t threads[] = {1};

    double *x = NULL;

    for (int i = 0; i < 1; ++i) {
        free(x);
        x = solution(temperatures, matrix, &params, threads[i], &(performance_stats[i]));
    }

    if (write_solution(x, x_nodes, y_nodes, SOLUTION_FILENAME) < 0) {
        fprintf(stderr, "solution writing error");
        return -1;
    }

    if (write_performance(threads, performance_stats, THREADS_NUM, PERFORMANCE_RESULTS_FILENAME) < 0) {
        fprintf(stderr, "performance writing error");
        return -1;
    }

    if (plot_solution(SOLUTION_FILENAME) < 0) {
        fprintf(stderr, "solution plotting error");
        return -1;
    }

    if (plot_performance(PERFORMANCE_RESULTS_FILENAME) < 0) {
        fprintf(stderr, "performance stats plotting error");
        return -1;
    }

    free(temperatures);
    for (int i = 0; i < nodes_amount; ++i) {
        free(matrix[i]);
    }
    free(matrix);
    free(x);

    return 0;
}
