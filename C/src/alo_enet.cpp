#include "blas_configuration.h"
#include "lasso_utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>

/** Compute the ALO leverage for the elastic net.
 * 
 * @param n The number of observations
 * @param p The number of active parameters
 * @param[in, out] XE A n x p matrix containing the active set
 * @param lde The leading dimension of E
 * @param sy The standard deviation of y times n
 * @param lambda The value of the regularizer lambda
 * @param alpha The value of the elastic net parameter alpha
 * @param has_intercept Whether an intercept was fit to the data
 * @param[out] h A vector of length n containing the leverage value for each observation.
 * @param[out] L If provided, a temporary array of size at least p * p to store the inner products.
 * 
 */
void alo_elastic_net(blas_size n, blas_size p, double* XE, blas_size lde,
                     double sy, double lambda, double alpha, bool has_intercept,
                     double* h, double* L) {
    bool alloc_l = false;

    if (!L) {
        L = (double*)blas_malloc(16, sizeof(double) * p * p);
    }

    double inv_sy = 1 / sy;
    double zero = 0;
    double one = 1;
    dsyrk("L", "T", &n, &p, &inv_sy, XE, &lde, &zero, L, &p);

    if (alpha != 1) {
        double offset = (1 - alpha) * lambda * inv_sy * inv_sy;

        for(blas_size i = 0; i < p; ++i) {
            L[i + p * i] += offset;
        }
    }

    if (has_intercept) {
        L[0] = 0;
    }

    int info;
    dpotrf("L", &p, L, &p, &info);
    dtrsm("R", "L", "T", "N", &n, &p, &one, L, &p, XE, &lde);

    for(blas_size i = 0; i < n; ++i) {
        double acc = 0;

        for(blas_size j = 0; j < n; ++j) {
            acc += XE[i * lde + j] * XE[i * lde + j];
        }

        h[i] = acc;
    }

    if (alloc_l) {
        blas_free(L);
    }
}

/** Compute the ALO leverage for the elastic net.
 * 
 * This function uses rectangular packed format for temporary storage.
 * 
 * @param L If provided, a temporary array of size at least p * (p + 1) / 2 to store inner products.
 */
void alo_elastic_net_rfp(blas_size n, blas_size p, double* XE, blas_size lde,
                         double sy, double lambda, double alpha, bool has_intercept,
                         double* h, double* L) {
    bool alloc_l = false;

    if (!L) {
        L = (double*)blas_malloc(16, sizeof(double) * p * (p + 1) / 2);
    }

    int ldl = p + (p % 2 == 0 ? 1 : 0);

    double inv_sy = 1 / sy;
    double zero = 0;
    double one = 1;

    dsfrk("N", "L", "T", &p, &n, &inv_sy, XE, &lde, &zero, L);

    if (alpha != 1) {
        double offset = (1 - alpha) * lambda * inv_sy * inv_sy;

        L[0] += offset;

        for(blas_size i = 1; i < (p + 1) / 2; ++i) {
            L[i + ldl * i] += offset;
            L[i + ldl * i - 1] += offset;
        }
    }

    if (has_intercept) {
        L[0] = 0;
    }

    int info;
    dpftrf("N", "L", &p, L, &info);
    dtfsm("N", "R", "L", "T", "N", &n, &p, &one, L, XE, &lde);

    for(blas_size i = 0; i < n; ++i) {
        double acc = 0;

        for(blas_size j = 0; j < n; ++j) {
            acc += XE[i * lde + j] * XE[i * lde + j];
        }

        h[i] = acc;
    }

    if (alloc_l) {
        blas_free(L);
    }
}

void copy_active_set(blas_size n, blas_size p, const double* A, blas_size lda, double* XE,
                     const std::vector<blas_size>& active_set) {
    for(blas_size i = 0; i < active_set.size(); ++i) {
        blas_size orig_column = active_set[i];
        std::copy(A + orig_column * lda, A + orig_column * lda + n, XE + i * n);
    }
}

double stddev(double* data, blas_size n) {
    double acc = 0;
    double acc2 = 0;

    for(blas_size i = 0; i < n; ++i) {
        acc += data[i];
        acc2 += data[i] * data[i];
    }

    acc2 -= acc * acc;
    acc2 /= n;

    return sqrt(acc2);
}

void enet_compute_alo_d(blas_size n, blas_size p, blas_size m, double* A, blas_size lda,
                        double* B, blas_size ldb, double* y, double* lambda, double alpha,
                        bool has_intercept,
                        double tolerance, double* alo, double* leverage) {
    blas_size max_active = max_active_set_size(m, p, B, ldb, tolerance);
    double* L = (double*)blas_malloc(16, max_active * (max_active + 1) * sizeof(double) / 2);
    double* XE = (double*)blas_malloc(16, max_active * n * sizeof(double));

    blas_size ld_leverage;
    bool alloc_leverage;
    if (leverage) {
        alloc_leverage = false;
        ld_leverage = n;
    }
    else {
        alloc_leverage = true;
        leverage = (double*)blas_malloc(16, n * sizeof(double));
        ld_leverage = 0;
    }

    double sy = n * stddev(y, n);

    for(blas_size i = 0; i < m; ++i) {
        std::vector<blas_size> current_index = find_active_set(p, B + ldb * i, tolerance);
        copy_active_set(n, p, A, lda, XE, current_index);

        alo_elastic_net_rfp(
            n, p, XE, n, sy, lambda[i], alpha, has_intercept,
            leverage + i * ld_leverage, L);
        
        compute_alo(n, p, A, lda, y, B + i * ldb, leverage + i * ld_leverage);
    }
}