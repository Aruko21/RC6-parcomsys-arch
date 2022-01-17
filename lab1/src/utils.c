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