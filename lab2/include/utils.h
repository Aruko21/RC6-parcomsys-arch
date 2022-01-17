#ifndef PLATEOPENMP_UTILS_H
#define PLATEOPENMP_UTILS_H

#include <stdio.h>
#include <stdlib.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "[CUDA ERROR]: %s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int print_matrix(double **A, size_t n);
int write_solution(double *x, int x_n, int y_n, const char *filename);
int write_performance(size_t *threads, double *performance_stats, size_t n, const char *filename);
int plot_solution(const char *solution_filename);
int plot_performance(const char *stats_filename);
void print_cuda_device_info();

#endif //PLATEOPENMP_UTILS_H
