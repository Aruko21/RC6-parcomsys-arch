#include "utils.h"

#include <stdlib.h>
#include <stdio.h>


int print_matrix(double **A, size_t n) {
    FILE *output = fopen("output_matr.dat", "w");
    if (!output) {
        fprintf(stderr, "output open error\n");
        return -1;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            fprintf(output, "%7.3lf ", A[i][j]);
        fprintf(output, "\n");
    }
    fclose(output);
    return 0;
}

int write_solution(double *x, int x_n, int y_n, const char *filename) {
    // Вывод
    FILE *output = fopen(filename, "w");
    if (!output) {
        fprintf(stderr, "'%s' open error\n", filename);
        return -1;
    }

    for (int i = 0; i < y_n; i++) {
        for (int j = 0; j < x_n; j++)
            fprintf(output, "%7.3lf ", x[i * x_n + j]);
        fprintf(output, "\n");
    }

    fclose(output);
    return 0;
}

int write_performance(size_t *threads, double *performance_stats, size_t n, const char *filename) {
    FILE *output = fopen(filename, "w");
    if (!output) {
        fprintf(stderr, "'%s' open error\n", filename);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        fprintf(output, "%llu %7.3lf", threads[i], performance_stats[i]);
        fprintf(output, "\n");
    }

    fclose(output);
    return 0;
}

int plot_solution(const char *solution_filename) {
    FILE *gp = fopen("solution_plot.gp", "w");

    if (!gp) {
        fprintf(stderr, "gnuplot open error\n");
        return -1;
    }

    fprintf(gp, "set pm3d map\n"
                "set pm3d interpolate 2,2\n"
                "set cbrange [0:100]\n"
                "set yrange [*:*] reverse\n"
                "set palette defined (0 \"#000000\", 1 \"#001aff\", 3 \"#00f2ff\", 5 \"#00ffaa\", 7 \"#f2ff00\", 10 \"#ff2200\")\n;"
                "set autoscale fix\n"
                "set cbtics 10\n"
                "splot '%s' matrix\n",
            solution_filename);
    fprintf(gp, "pause -1\n");
    fclose(gp);

    system("gnuplot solution_plot.gp");
    return 0;
}

int plot_performance(const char *stats_filename) {
    FILE *gp = fopen("perf_gnuplot.gp", "w");

    if (!gp) {
        fprintf(stderr, "gnuplot open error\n");
        return -1;
    }

    fprintf(gp, "set term wxt title 'Performance statistics t(N)'\n"
                "set xlabel 'Number of threads (N)'\n"
                "set ylabel 'Solution time (t)'\n"
                "set key right bottom\n"
                "set grid\n"
                "plot '%s' using 1:2 with l lc rgb 'blue' lt 1 lw 1.5 title 't(N)'\n",
            stats_filename);
    fprintf(gp, "pause -1\n");

    fclose(gp);
    system("gnuplot perf_gnuplot.gp");
    return 0;
}

void print_cuda_device_info() {
    int device_count = 0;
    cudaDeviceProp device_prop;

    // Сколько устройств CUDA установлено на PC.
    cudaGetDeviceCount(&device_count);

    printf("Device count: %d\n\n", device_count);

    for (int i = 0; i < device_count; i++)
    {
        //Получаем информацию об устройстве
        cudaGetDeviceProperties(&device_prop, i);

        //Выводим иформацию об устройстве
        printf("Device name: %s\n", device_prop.name);
        printf("Total global memory: %d\n", device_prop.totalGlobalMem);
        printf("Shared memory per block: %d\n", device_prop.sharedMemPerBlock);
        printf("Registers per block: %d\n", device_prop.regsPerBlock);
        printf("Warp size: %d\n", device_prop.warpSize);
        printf("Memory pitch: %d\n", device_prop.memPitch);
        printf("Max threads per block: %d\n", device_prop.maxThreadsPerBlock);

        printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
               device_prop.maxThreadsDim[0],
               device_prop.maxThreadsDim[1],
               device_prop.maxThreadsDim[2]);

        printf("Max grid size: x = %d, y = %d, z = %d\n",
               device_prop.maxGridSize[0],
               device_prop.maxGridSize[1],
               device_prop.maxGridSize[2]);

        printf("Clock rate: %d\n", device_prop.clockRate);
        printf("Total constant memory: %d\n", device_prop.totalConstMem);
        printf("Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
        printf("Texture alignment: %d\n", device_prop.textureAlignment);
        printf("Device overlap: %d\n", device_prop.deviceOverlap);
        printf("Multiprocessor count: %d\n", device_prop.multiProcessorCount);

        printf("Kernel execution timeout enabled: %s\n",
               device_prop.kernelExecTimeoutEnabled ? "true" : "false");
    }
}