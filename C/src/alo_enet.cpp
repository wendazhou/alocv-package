#include "alocv/alo_enet.h"
#include "blas_configuration.h"
#include "lasso_utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>

#include "gram_utils.hpp"

namespace {

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

}


void compute_fitted(blas_size n, blas_size k, const double* XE,
                    const double* beta, double a0, bool has_intercept,
                    const std::vector<blas_size>& index, double* y_fitted) {
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
            std::fill(y_fitted, y_fitted + n, 0.0);
        } else {
            copy_active_set(n, A, lda, has_intercept, current_index, std::vector<blas_size>(), XE, n);
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