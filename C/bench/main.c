#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"

#include "alocv/alo_lasso.h"
#include "alocv/alo_enet.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define NOMINMAX
#include "windows.h"
#include "malloc.h"

double get_current_time() {
    static double performance_frequency = 0;

    if(performance_frequency == 0) {
        LARGE_INTEGER perf_freq;
        QueryPerformanceFrequency(&perf_freq);
        performance_frequency = (double)perf_freq.QuadPart;
    }

    LARGE_INTEGER performance_count;
    QueryPerformanceCounter(&performance_count);

    return (double)performance_count.QuadPart / performance_frequency;
}

inline void* aligned_alloc(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}

#define aligned_free _aligned_free

#else
#include "time.h"

double get_current_time() {
    struct timespec res;
    clock_gettime(CLOCK_MONOTONIC, &res);

    return (double)res.tv_sec + ((double)res.tv_nsec / 1e9);
}

#define aligned_free free

#endif

typedef enum {
    AloMethodLasso,
    AloMethodEnet,
    AloMethodFishnet,
    AloMethodLognet,
} AloMethod;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Please pass in input file.\n");
        return 0;
    }

    AloMethod method;

    if(!strcmp("lasso", argv[1])) {
        method = AloMethodLasso;
    }
    else if (!strcmp("enet", argv[1])) {
        method = AloMethodEnet;
    }
    else if (!strcmp("fishnet", argv[1])) {
        method = AloMethodFishnet;
    }
    else if (!strcmp("lognet", argv[1])) {
        method = AloMethodLognet;
    }
    else {
        printf("Unknow method specified: %s", argv[1]);
        return 1;
    }

    int32_t n, p, m;

    FILE* file = fopen(argv[2], "rb");

    fread(&n, sizeof(n), 1, file);
    fread(&p, sizeof(p), 1, file);
    fread(&m, sizeof(m), 1, file);

    double* A = aligned_alloc(16, n * p * sizeof(double));
    double* B = aligned_alloc(16, p * m * sizeof(double));
    double* y = aligned_alloc(16, n * sizeof(double));
    double* lambda = aligned_alloc(16, m * sizeof(double));

    double* alo_result = aligned_alloc(16, m * sizeof(double));

    printf("Reading data: n=%d, p=%d, m=%d\n", n, p, m);

    fread(A, sizeof(double), n * p, file);
    fread(B, sizeof(double), p * m, file);
    fread(y, sizeof(double), n, file);
    fread(lambda, sizeof(double), m, file);

    printf("Loaded data. Performing initial run.\n");

    double start = get_current_time();

    switch(method) {
    case 0:
        printf("Computing ALO for method: LASSO\n");
        lasso_compute_alo_d(n, p, m, A, n, B, p, y, 1, 1e-5, alo_result, NULL);
        break;
    case 1:
        printf("Computing ALO for method: Elastic-Net\n");
        enet_compute_alo_d(n, p, m, A, n, B, p, y, NULL, lambda, 0.9, 0, GlmFamilyGaussian, 0, 1e-5, alo_result,
                           NULL, NULL, NULL);
        break;
    case 2:
        printf("Computing ALO for method: Poisson Elastic-Net\n");
        enet_compute_alo_d(n, p, m, A, n, B, p, y, NULL, lambda, 1.0, 0, GlmFamilyPoisson, 1, 1e-5, alo_result,
                           NULL, NULL, NULL);
        break;
    case 3:
        printf("Computing ALO for method: Logistic Elastic-Net\n");
        enet_compute_alo_d(n, p, m, A, n, B, p, y, NULL, lambda, 0.9, 0, GlmFamilyLogit, 1, 1e-5, alo_result,
                           NULL, NULL, NULL);
        break;
    default:
        printf("Unknown method");
        return 1;
    }

    double end = get_current_time();

    printf("Done in %f seconds\n", end - start);

    double alo_min = 0.0;
    double alo_max = INFINITY;
    double alo_mean = 0.0;

    for (int i = 0; i < m; ++i) {
        alo_mean += alo_result[i];
        alo_min = fmin(alo_min, alo_result[i]);
        alo_max = fmax(alo_max, alo_result[i]);
    }

    alo_mean /= m;

    printf("Stastistics for computed ALO: min=%g; mean=%g; max=%g", alo_min, alo_mean, alo_max);

    aligned_free(alo_result);
    aligned_free(y);
    aligned_free(B);
    aligned_free(A);

    return 0;
}
