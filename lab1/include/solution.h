#ifndef PLATEOPENMP_SOLUTION_H
#define PLATEOPENMP_SOLUTION_H

#include <stdlib.h>

struct solution_params {
    double time_end;
    double step;
    double height;
    double width;
    double right_border;
    double top_border;
    double temp_init;
    double alpha;
};

double *gauss(double **A_matrix, const double *y, size_t nodes_num, size_t threads_num);
void initialise(double **vector, double ***matrix, struct solution_params *params);
double *solution(double *y, double **A, struct solution_params *params, size_t threads_num, double *performance_stats);

#endif //PLATEOPENMP_SOLUTION_H
