#include "alocv/alo_enet.h"
#include "blas_configuration.h"
#include "lasso_utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>


namespace {

/*! Computes the Gram matrix of the given dataset in RFP format.
 * 
 */
void compute_gram_rfp(blas_size n, blas_size p, const double* XE, blas_size lde, double* L) {
    const double one = 1;
    const double zero = 0;

#ifndef ALOCV_LAPACK_NO_RFP
    dsfrk("N", "L", "T", &p, &n, &one, XE, &lde, &zero, L);
#else
    const bool is_odd = p % 2;
    const blas_size p2 = p / 2;
    const blas_size p1 = p - p2;

    dsyrk("L", "T", &p1, &n, &one, XE, &lde, &zero, L + (is_odd ? 0 : 1), &p);
    dsyrk("U", "T", &p2, &n, &one, XE + p1, &lde, &zero, L + (is_odd ? p : 0), &p);
    dgemm("T", "N", &p2, &p1, &n, &one, XE + p1, &lde, XE, &lde, &zero, L + p1, &p);
#endif
}


/*! Computes the Cholesky decomposition of the matrix in RFP format. */
int compute_cholesky_rfp(blas_size p, double* L) {
    int info;

#ifndef ALOCV_LAPACK_NO_RFP
    dpftrf("N", "L", &p, L, &info);
#else
    const bool is_odd = p % 2;
    const blas_size p2 = p / 2;
    const blas_size p1 = p - p2;

    const blas_size ldl = is_odd ? p : p + 1;

    const blas_size o1 = is_odd ? 0 : 1;
    const blas_size o2 = is_odd ? p1 : p1 + 1;
    const blas_size o3 = is_odd ? p : 0;

    dpotrf("L", &p1, L + o1, &ldl, &info);

    if(info) return info;

    const double one = 1;
    const double neg_one = -1;

    dtrsm("R", "L", "T", "N", &p2, &p1, &one, L + o1, &ldl, L + o2, &ldl);
    dsyrk("U", "N", &p2, &p1, &neg_one, L + o2, &ldl, &one, L + o3, &ldl);
    dpotrf("U", &p2, L + o3, &ldl, &info);
#endif
    return info;
}

int solve_triangular_rfp(blas_size n, blas_size p, const double* L, double* XE, blas_size lde) {
    const double one = 1;

#ifndef ALOCV_LAPACK_NO_RFP
    dtfsm("N", "R", "L", "T", "N", &n, &p, &one, L, XE, &lde);
#else
    const bool is_odd = p % 2;
    const blas_size p2 = p / 2;
    const blas_size p1 = p - p2;

    const blas_size ldl = is_odd ? p : p + 1;

    const double neg_one = -1;

    dtrsm("R", "L", "T", "N", &n, &p1, &one, L + (is_odd ? 0 : 1), &ldl, XE, &lde);
    dgemm("N", "T", &n, &p1, &p2, &neg_one, XE, &lde, L + p1 + (is_odd ? 0 : 1), &ldl, &one, XE + p1 * lde, &lde);
    dtrsm("R", "U", "N", "N", &n, &p2, &one, L + p, &ldl, XE + p1 * lde, &lde);
#endif
}

}


/** Compute the ALO leverage for the elastic net.
 * 
 * @param n The number of observations
 * @param p The number of active parameters
 * @param[in, out] XE A n x p matrix containing the active set
 * @param lde The leading dimension of E
 * @param lambda The value of the regularizer lambda
 * @param alpha The value of the elastic net parameter alpha
 * @param has_intercept Whether an intercept was fit to the data
 * @param[out] h A vector of length n containing the leverage value for each observation.
 * @param[out] L If provided, a temporary array of size at least p * p to store the inner products.
 * 
 */
void alo_elastic_net(blas_size n, blas_size p, double* XE, blas_size lde,
                     double lambda, double alpha, bool has_intercept,
                     double* h, double* L) {
    bool alloc_l = false;

    if (!L) {
        L = (double*)blas_malloc(16, sizeof(double) * p * p);
    }

    double zero = 0;
    double one = 1;
    dsyrk("L", "T", &p, &n, &one, XE, &lde, &zero, L, &p);

    if (alpha != 1) {
        double offset = (1 - alpha) * lambda;

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
        h[i] = ddot(&p, XE + i, &n, XE + i, &n);
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
                         double lambda, double alpha, bool has_intercept,
                         double* h, double* L) {
    bool alloc_l = false;

    if (!L) {
        L = (double*)blas_malloc(16, sizeof(double) * p * (p + 1) / 2);
    }

    int ldl = p + (p % 2 == 0 ? 1 : 0);

    double zero = 0;
    double one = 1;

    compute_gram_rfp(n, p, XE, lde, L);

    if (alpha != 1) {
        double offset = (1 - alpha) * lambda;

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
    compute_cholesky_rfp(p, L);
    solve_triangular_rfp(n, p, L, XE, lde);

    for(blas_size i = 0; i < n; ++i) {
        h[i] = ddot(&p, XE + i, &n, XE + i, &n);
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

double stddev(const double* data, blas_size n) {
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

void enet_compute_alo_d(blas_size n, blas_size p, blas_size m, const double* A, blas_size lda,
                        const double* B, blas_size ldb, const double* y, const double* lambda, double alpha,
                        int has_intercept,
                        double tolerance, double* alo, double* leverage) {
    blas_size max_active = max_active_set_size(m, p, B, ldb, tolerance);

    const std::size_t l_size = max_active * (max_active + 1) / 2;

    double* L = (double*)blas_malloc(16, l_size * sizeof(double));
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

    for(blas_size i = 0; i < m; ++i) {
        std::vector<blas_size> current_index = find_active_set(p, B + ldb * i, tolerance);

        if(current_index.empty()) {
            // no selected variables
            std::fill(leverage + i * ld_leverage, leverage + i * ld_leverage + n, 0.0);
        } else {
            copy_active_set(n, p, A, lda, XE, current_index);
            alo_elastic_net_rfp(
                n, current_index.size(), XE, n, lambda[i], alpha, has_intercept,
                leverage + i * ld_leverage, L);
        }
        
        alo[i] = compute_alo(n, p, A, lda, y, B + i * ldb, leverage + i * ld_leverage);
    }

    if(alloc_leverage) {
        blas_free(leverage);
    }

    blas_free(L);
    blas_free(XE);
}