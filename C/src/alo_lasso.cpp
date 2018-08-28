#include "alocv/alo_lasso.h"
#include "alocv/cholesky_utils.h"
#include "blas_configuration.h"

#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>
#include <cmath>
#include <cstdio>


void lasso_update_cholesky_w_d(blas_size n, double* A, blas_size lda,
                               double* L, blas_size ldl,
                               double* W, blas_size ldw, 
                               blas_size len_index, blas_size* index,
                               blas_size len_index_new, blas_size* index_new) {
    // First compute the columns to add and remove from the matrix to update it.
    std::vector<blas_size> start_index(index, index + len_index);
    std::vector<blas_size> end_index(index_new, index_new + len_index_new);

    std::vector<blas_size> active_index(start_index);

    std::sort(start_index.begin(), start_index.end());
    std::sort(end_index.begin(), end_index.end());

    std::vector<blas_size> index_added;
    std::vector<blas_size> index_removed;

    std::set_difference(
        end_index.begin(), end_index.end(),
        start_index.begin(), start_index.end(),
        std::back_inserter(index_added));
    
    std::set_difference(
        start_index.begin(), start_index.end(),
        end_index.begin(), end_index.end(),
        std::back_inserter(index_removed));
    
    // Delete all unnecessary columns first.
    for(auto i: index_removed) {
        auto it = std::find(active_index.begin(), active_index.end(), i);
        auto loc = std::distance(active_index.begin(), it);

        cholesky_delete_inplace_d(active_index.size(), loc, L, ldl);
        active_index.erase(it);
    }
    
    {
        std::size_t col_w = 0;

        // Once we have deleted required columns, we are only appending to the
        // end. This is a good time to construct the W matrix.
        // First add all the existing indices.
        for (auto col_a: active_index) {
            std::copy(A + col_a * lda, A + col_a * lda + n, W + ldw * col_w);
            col_w += 1;
        }

        // Add the new indices.
        for (auto col_a : index_added) {
            std::copy(A + col_a * lda, A + col_a * lda + n, W + ldw * col_w);
            col_w += 1;
        }
    }

    // Precompute the border of the matrix we are appending.
    blas_size num_existing = active_index.size();
    blas_size num_added = index_added.size();
    blas_size num_total = num_existing + num_added;
    double one_d = 1.0;
    double zero_d = 0.0;

    // Compute the covariance of the added columns. This places it in the lower
    // half of the existing decomposition L.
    dgemm("T", "N", &num_added, &num_existing, &n, &one_d, W + num_existing * ldw, &ldw, W, &ldw, &zero_d, L + num_existing, &ldl);
    dsyrk("L", "N", &num_added, &n, &one_d, W + num_existing * ldw, &ldw, &zero_d, L + num_existing * ldl + num_existing, &ldl);

    // Append all necessary indices to reach the desired state.
    cholesky_append_inplace_multiple_d(active_index.size(), index_added.size(), L, ldl);
    std::copy(index_added.begin(), index_added.end(), std::back_inserter(active_index));

    // index_new contains the corresponding set of indices.
    assert(active_index.size() == len_index_new);
    std::copy(active_index.begin(), active_index.end(), index_new);
}


void lasso_compute_leverage_cholesky_d(blas_size n, blas_size k, double* W, blas_size ldw,
                                       double* L, blas_size ldl, double* leverage) {
    double one_d = 1.0;
    dtrsm("R", "L", "T", "N", &n, &k, &one_d, L, &ldl, W, &n);

    for(blas_size i = 0; i < n; ++i) {
        leverage[i] = ddot(&k, W + i, &n, W + i, &n);
    }
}

/*! For a given coefficient set beta, finds the active set by a magnitude-based rule.
 *
 *  @param[in] p The number of coefficients.
 *  @param[in] beta A pointer to the coefficients.
 *  @param[in] tolerance The tolerance to determine which values are 0.
 */
std::vector<blas_size> find_active_set(blas_size p, double* beta, double tolerance) {
    std::vector<blas_size> result;

    for(blas_size i = 0; i < p; ++i) {
        if (std::abs(beta[i]) > tolerance) {
            result.push_back(i);
        }
    }

    return result;
}


/*! Finds the largest active set in the given set of solutions.
 */
blas_size max_active_set_size(blas_size num_tuning, blas_size p, double* B, blas_size ldb, double tolerance) {
    blas_size max_size = 0;

    for(blas_size i = 0; i < num_tuning; ++ i) {
        blas_size current_size = std::count_if(B + ldb * i, B + ldb * i + p, [=](double x) { return std::abs(x) > tolerance; });
        max_size = std::max(max_size, current_size);
    }

    return max_size;
}


double compute_alo(blas_size n, blas_size p, double* A, blas_size lda, double* y,
                   double* beta, double* leverage, blas_size incl) {
    double* temp = static_cast<double*>(blas_malloc(16, n * sizeof(double)));

    // temp = y
    std::copy(y, y + n, temp);

    double one_d = 1.0;
    double min_one_d = -1.0;
    blas_size one_i = 1;

    // temp = X * beta - y
    dgemv("N", &n, &p, &one_d, A, &lda, beta, &one_i, &min_one_d, temp, &one_i);

    double acc = 0;

    for(blas_size i = 0; i < n; ++i) {
        double res = temp[i] / (1 - leverage[incl * i]);
        acc += res * res;
    }

    blas_free(temp);

    return acc / n;
}

void compute_cholesky(blas_size n, blas_size k, double* W, blas_size ldw, double* L, blas_size ldl) {
    blas_size one = 1;
    double one_d = 1.0;
    double zero_d = 0.0;
    blas_size info;

#ifdef USE_MKL
    dgemmt("L", "T", "N", &k, &n, &one_d, W, &ldw, W, &ldw, &zero_d, L, &ldl);
#else
    dgemm("T", "N", &k, &k, &n, &one_d, W, &ldw, W, &ldw, &zero_d, L, &ldl);
#endif
    dpotrf("L", &k, L, &ldl, &info);
}


/*! Update the copy of the data in the active set we maintain.
 *
 *  In order to efficiently compute the cholesky decomposition and the residuals,
 *  we maintain a copy of the active set.
 */
void create_w(blas_size n, double* W, blas_size ldw, double* A, blas_size lda, std::vector<blas_size> const& current_index) {
    // The index is updated by removing some existing columns, and appending new columns
    // at the end of the representation.

    for(int i = 0; i < current_index.size(); ++i) {
        std::copy(A + current_index[i] * lda, A + current_index[i] * lda + n, W + i * ldw);
    }
}


void lasso_compute_alo_d(blas_size n, blas_size p, blas_size m, double* A, blas_size lda,
                         double* B, blas_size ldb, double* y, blas_size incy, double tolerance,
                         double* alo, double* leverage) {
    // Allocate necessary structures
    blas_size max_active = max_active_set_size(m, p, B, ldb, tolerance);

    double* L = static_cast<double*>(blas_malloc(16, max_active * max_active * sizeof(double)));
    blas_size L_active = 0;
    blas_size ldl = max_active;

    double* W = static_cast<double*>(blas_malloc(16, n * max_active * sizeof(double)));
    blas_size ldw = n;

    std::vector<blas_size> active_index;

    blas_size ld_leverage;
    bool alloc_leverage;
    if (leverage) {
        alloc_leverage = false;
        ld_leverage = n;
    }
    else {
        alloc_leverage = true;
        leverage = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
        ld_leverage = 0;
    }

    for(blas_size i = 0; i < m; ++i) {
        std::vector<blas_size> current_index = find_active_set(p, B + ldb * i, tolerance);

        auto num_active = current_index.size();

        if (num_active == 0) {
            // no active set, reset current path.
            double zero_leverage = 0.0;

            if (!alloc_leverage) {
                // We are using leverage as an output argument
                // we are thus required to set the actual leverage values.
                std::fill(leverage + ld_leverage * i, leverage + ld_leverage * (i + 1), 0.0);
            }

            alo[i] = compute_alo(n, p, A, lda, y, B + ldb * i, &zero_leverage, 0);
            L_active = 0;
            continue;
        }

        // First, we need to make sure our Cholesky decomposition is up to date.
        if (L_active > 0) {
            // update our cholesky decomposition
            lasso_update_cholesky_w_d(n, A, lda, L, ldl, W, ldw,
                active_index.size(), active_index.data(), current_index.size(), current_index.data());
        }
        else {
            // no existing cholesky decomposition, allocate memory and compute a new one.
            create_w(n, W, ldw, A, lda, current_index);
            compute_cholesky(n, num_active, W, ldw, L, ldl);
        }

        // update our current copy of the active set.
        L_active = num_active;

        // update the active index
        std::swap(active_index, current_index);

        // Now that we have the Cholesky decomposition, let's compute the leverage values.
        if (num_active >= n) {
            // special case where the active set is of the same size as the
            // number of observations. The ALO estimate of risk is infinite.

            if (!alloc_leverage) {
                // if we are outputting leverage values we should fill this too.
                std::fill(leverage + ld_leverage * i, leverage + ld_leverage * (i + 1), 1.0);
            }

            alo[i] = INFINITY;
            continue;
        }

        // compute the leverage value.
        lasso_compute_leverage_cholesky_d(n, num_active, W, ldw, L, ldl, leverage + ld_leverage * i);
        // compute the current ALO vlue.
        alo[i] = compute_alo(n, p, A, lda, y, B + ldb * i, leverage + ld_leverage * i, 1);
    }

    // free all the buffers
    blas_free(L);

    if (alloc_leverage) {
        blas_free(leverage);
    }
}