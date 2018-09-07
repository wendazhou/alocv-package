#include "alocv/alo_enet.h"
#include "blas_configuration.h"
#include "lasso_utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>

#include "gram_utils.hpp"


/*! Computes the Gram matrix of the given dataset in RFP format.
 * 
 */
void compute_gram(blas_size n, blas_size p, const double* XE, blas_size lde, double* L, SymmetricFormat format) {
    const double one = 1;
    const double zero = 0;

    if(format == SymmetricFormat::Full) {
        dsyrk("L", "T", &p, &n, &one, XE, &lde, &zero, L, &p);
        return;
    }

#ifndef ALOCV_LAPACK_NO_RFP
    dsfrk("N", "L", "T", &p, &n, &one, XE, &lde, &zero, L);
#else
    const bool is_odd = p % 2;
    const blas_size p2 = p / 2;
    const blas_size p1 = p - p2;
    const blas_size ldl = is_odd ? p : p + 1;

    dsyrk("L", "T", &p1, &n, &one, XE, &lde, &zero, L + (is_odd ? 0 : 1), &ldl);
    dsyrk("U", "T", &p2, &n, &one, XE + p1 * lde, &lde, &zero, L + (is_odd ? p : 0), &ldl);
    dgemm("T", "N", &p2, &p1, &n, &one, XE + p1 * lde, &lde, XE, &lde, &zero, L + p1 + (is_odd ? 0 : 1), &ldl);
#endif
}

/*! Computes the Cholesky decomposition of the matrix in RFP format. */
int compute_cholesky(blas_size p, double* L, SymmetricFormat format) {
    int info;

    if(format == SymmetricFormat::Full) {
        dpotrf("L", &p, L, &p, &info);
        return info;
    }

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

void solve_triangular(blas_size n, blas_size p, const double* L, double* XE, blas_size lde, SymmetricFormat format) {
    const double one = 1;

    if(format == SymmetricFormat::Full) {
        dtrsm("R", "L", "T", "N", &n, &p, &one, L, &p, XE, &lde);
        return;
    }

#ifndef ALOCV_LAPACK_NO_RFP
    dtfsm("N", "R", "L", "T", "N", &n, &p, &one, L, XE, &lde);
#else
    const bool is_odd = p % 2;
    const blas_size p2 = p / 2;
    const blas_size p1 = p - p2;

    const blas_size ldl = is_odd ? p : p + 1;

    const double neg_one = -1;

    dtrsm("R", "L", "T", "N", &n, &p1, &one, L + (is_odd ? 0 : 1), &ldl, XE, &lde);
    dgemm("N", "T", &n, &p2, &p1, &neg_one, XE, &lde, L + p1 + (is_odd ? 0 : 1), &ldl, &one, XE + p1 * lde, &lde);
    dtrsm("R", "U", "N", "N", &n, &p2, &one, L + (is_odd ? p : 0), &ldl, XE + p1 * lde, &lde);
#endif
}

namespace {


/*! Adds the given value to the diagonal of the matrix.
 *
 * @param p The size of the matrix.
 * @param[in, out] L The matrix in the given format.
 * @param value The value to add to the diagonal.
 * @param skip_first If true, the value is not added to the first element.
 * 
 */
void offset_diagonal(blas_size p, double* L, double value, bool skip_first, SymmetricFormat format) {
    if(format == SymmetricFormat::Full) {
        for(int i = skip_first; i < p; ++i) {
            L[i * p + i] += value;
        }

        return;
    } else {
        if(p % 2 == 1) {
            // odd case

            if(!skip_first) {
                L[0] += value;
            }

            for(blas_size i = 1; i < (p + 1) / 2; ++i) {
                L[i + p * i] += value;
                L[i + p * i - 1] += value;
            }
        } else {
            // even case
            L[0] += value;

            if(!skip_first) {
                L[1] += value;
            }

            for(blas_size i = 1; i < p / 2; ++i) {
                L[i + p * i] += value;
                L[i + p * i + 1] += value;
            }
        }
    }
}

blas_size sym_num_elements(blas_size p, SymmetricFormat format) {
    if(format == SymmetricFormat::Full) {
        return p * p;
    }
    else {
        return p * (p + 1) / 2;
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
 * @param format The format to use for storing intermediate products.
 * 
 */
void alo_elastic_net_rfp(blas_size n, blas_size p, double* XE, blas_size lde,
                         double lambda, double alpha, bool has_intercept,
                         double* h, double* L, SymmetricFormat format) {
    bool alloc_l = false;
    blas_size p_effective = p + (has_intercept ? 1 : 0);

    if (!L) {
        L = (double*)blas_malloc(16, sizeof(double) * sym_num_elements(p_effective, format));
    }

    double one = 1;

    compute_gram(n, p_effective, XE, lde, L, format);

    if (alpha != 1) {
        double offset = (1 - alpha) * lambda;
        offset_diagonal(p_effective, L, offset, has_intercept, format);
    }

    int info;
    compute_cholesky(p_effective, L, format);
    solve_triangular(n, p_effective, L, XE, lde, format);

    for(blas_size i = 0; i < n; ++i) {
        h[i] = ddot(&p_effective, XE + i, &n, XE + i, &n);
    }

    if (alloc_l) {
        blas_free(L);
    }
}

/*! Compute fitted response in link space.
 *
 * @param n The number of observations
 * @param k The number of non-zero parameters.
 * @param XE[in] A matrix containing the active set predictors
 * @param beta[in] A vector containing the parameters.
 * @param a0 A double representing the value of the intercept. Ignored if no intercept.
 * @param has_intercept If true, signifies that an intercept was fitted.
 * @param index A list containing the indices of the active set.
 * @param y_fitted[out] A vector of length n which will be filled with the fitted link space result.
 * 
 */
void compute_fitted(blas_size n, blas_size k, const double* XE,
                    const double* beta, double a0, bool has_intercept,
                    const std::vector<blas_size>& index,
                    double* y_fitted) {
    double* beta_active = (double*)blas_malloc(16, (index.size() + has_intercept) * sizeof(double));

    if(has_intercept) {
        beta_active[0] = a0;
        k += 1;
    }

    for(int i = 0; i < index.size(); ++i) {
        beta_active[i + has_intercept] = beta[index[i]];
    }

    double zero = 0.0;
    double one = 1.0;
    blas_size one_i = 1;

    dgemv("N", &n, &k, &one, XE, &n, beta_active, &one_i, &zero, y_fitted, &one_i);

    blas_free(beta_active);
}


double compute_alo_fitted(blas_size n, const double* y, const double* y_fitted, const double* leverage) {
    double acc = 0;

    for(blas_size i = 0; i < n; ++i) {
        double res = (y[i] - y_fitted[i]) / (1 - leverage[i]);
        acc += res * res;
    }

    return acc / n;
}

}

void copy_active_set(blas_size n, blas_size p, const double* A, blas_size lda, double* XE,
                     const std::vector<blas_size>& active_set, bool has_intercept) {
    if(has_intercept) {
        // First column is intercept column if present.
        std::fill(XE, XE + n, 1.0);
    }
    
    for(blas_size i = 0; i < active_set.size(); ++i) {
        blas_size orig_column = active_set[i];
        std::copy(A + orig_column * lda, A + orig_column * lda + n,
                  XE + (i + has_intercept) * n);
    }
}

void enet_compute_alo_d(blas_size n, blas_size p, blas_size m, const double* A, blas_size lda,
                        const double* B, blas_size ldb, const double* y, const double* a0,
                        const double* lambda, double alpha,
                        int has_intercept, int use_rfp,
                        double tolerance, double* alo, double* leverage) {
    SymmetricFormat format = use_rfp ? SymmetricFormat::RFP : SymmetricFormat::Full;

    blas_size max_active = max_active_set_size(m, p, B, ldb, tolerance) + (has_intercept ? 1 : 0);
    const std::size_t l_size = sym_num_elements(max_active, format);

    double* y_fitted = (double*)blas_malloc(16, n * sizeof(double));
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

        if(current_index.empty() && !has_intercept) {
            // no selected variables
            std::fill(leverage + i * ld_leverage, leverage + i * ld_leverage + n, 0.0);
        } else {
            copy_active_set(n, p, A, lda, XE, current_index, has_intercept);
            compute_fitted(n, current_index.size(), XE, B + ldb * i, has_intercept ? a0[i] : 0.0,
                           has_intercept, current_index, y_fitted);
            alo_elastic_net_rfp(
                n, current_index.size(), XE, n, lambda[i], alpha, has_intercept,
                leverage + i * ld_leverage, L, format);
        }

        alo[i] = compute_alo_fitted(n, y, y_fitted, leverage + i * ld_leverage);
    }

    if(alloc_leverage) {
        blas_free(leverage);
    }

    blas_free(y_fitted);
    blas_free(L);
    blas_free(XE);
}