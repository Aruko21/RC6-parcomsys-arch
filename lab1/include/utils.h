#ifndef PLATEOPENMP_UTILS_H
#define PLATEOPENMP_UTILS_H

#include <stdlib.h>

int print_matrix(double **A, size_t n);
int write_solution(double *x, int x_n, int y_n, const char *filename);
int write_performance(size_t *threads, double *performance_stats, size_t n, const char *filename);
int plot_solution(const char *solution_filename);
int plot_performance(const char *stats_filename);

#endif //PLATEOPENMP_UTILS_H
