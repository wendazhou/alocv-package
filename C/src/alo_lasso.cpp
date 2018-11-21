#include "alocv/alo_lasso.h"
#include "lasso_utils.h"
#include "alocv/cholesky_utils.h"
#include "blas_configuration.h"

#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>
#include <cmath>
#include <cstdio>

/*! Copies over the active predictors to a packed matrix.
 *
 * @param n The number of observations
 * @param[in] A The matrix of all predictors.
 * @param lda The leading dimension of A.
 * @param has_intercept Whether to add an intercept column.
 * @param index_active The index of currently active predictors.
 * @param index_added Another optional vector of indices to append to the active predictors.
 * @param[out] W A matrix to copy the predictors to.
 * @param ldw The leading dimension of W, must be at least n.
 * 
 * 
 */
void copy_active_set(blas_size n, const double* A, blas_size lda, bool has_intercept,
                   std::vector<blas_size> const& index_active, std::vector<blas_size> const& index_added,
                   double* W, blas_size ldw) {
    std::size_t col_w = 0;

    if(has_intercept) {
        std::fill(W, W + n, 1.0);
        col_w += 1;
    }

    for (auto col_a : index_active) {
        std::copy(A + col_a * lda, A + col_a * lda + n, W + ldw * col_w);
        col_w += 1;
    }

    // Add the new indices.
    for (auto col_a : index_added) {
        std::copy(A + col_a * lda, A + col_a * lda + n, W + ldw * col_w);
        col_w += 1;
    }
}

namespace {
/*! Utility function to update the Cholesky decomposition along the lasso path.
 *
 * An essential component in computing the leverage values for the LASSO estimator is to compute the
 * inverse of the covariance of the active set. We do this by maintaining the Cholesky decomposition
 * of the covariance of the active set along the solution path, and update this decomposition iteratively
 * as we go down the solution path.
 * 
 * For performance reasons, we only append new active coordinates at the end of the decomposition.
 * For this reason, we need to maintain the corresponding ordering of the elements in our decomposition.
 * 
 * @param[in] n The number of observations (or rows) of A
 * @param[in] A The regression matrix
 * @param[in] lda The leading dimension of A
 * @param[in,out] L The Cholesky decomposition of the current active set in lower triangular form.
 * @param[in] ldl The leading dimension of L
 * @param[out] W The updated regression matrix restricted to the active set.
 * @param[in] ldw The leading dimension of W.
 * @param[in] len_index The size of the current active set.
 * @param[in] index The indices of the columns that are currently active.
 * @param[in] len_index_new The size of the new active set.
 * @param[in,out] index_new The indices of the new active set. This will be re-ordered to represent the new
 *      active set in the order in which they appear in the decomposition.
 *
 */
void lasso_update_cholesky_w_d(blas_size n, const double* A, blas_size lda,
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
        auto loc = static_cast<blas_size>(std::distance(active_index.begin(), it));

        cholesky_delete_inplace_d(static_cast<blas_size>(active_index.size()), loc, L, ldl);
        active_index.erase(it);
    }

    copy_active_set(n, A, lda, false, active_index, index_added, W, ldw);
    
    // Precompute the border of the matrix we are appending.
    blas_size num_existing = static_cast<blas_size>(active_index.size());
    blas_size num_added = static_cast<blas_size>(index_added.size());
    blas_size num_total = num_existing + num_added;
    double one_d = 1.0;
    double zero_d = 0.0;

    // Compute the covariance of the added columns. This places it in the lower
    // half of the existing decomposition L.
    dgemm("T", "N", &num_added, &num_existing, &n, &one_d, W + num_existing * ldw, &ldw, W, &ldw, &zero_d, L + num_existing, &ldl);
    dsyrk("L", "T", &num_added, &n, &one_d, W + num_existing * ldw, &ldw, &zero_d, L + num_existing * ldl + num_existing, &ldl);

    // Append all necessary indices to reach the desired state.
    cholesky_append_inplace_multiple_d(static_cast<blas_size>(active_index.size()), static_cast<blas_size>(index_added.size()), L, ldl);
    std::copy(index_added.begin(), index_added.end(), std::back_inserter(active_index));

    // index_new contains the corresponding set of indices.
    assert(active_index.size() == len_index_new);
    std::copy(active_index.begin(), active_index.end(), index_new);
}


/*! Utility function to compute the leverage value from the cholesky decomposition maintained by the algorithm.
 *
 * @param[in] n The number of observations (or rows of A).
 * @param[in] k The size of the active set.
 * @param[in] W The regression matrix on the active set.
 * @param[in] ldw The leading dimension of W.
 * @param[in] L The Cholesky decomposition of the covariance of the active set.
 * @param[in] ldl The leading dimension of L.
 * @param[out] leverage The computed leverage values.
 */
void lasso_compute_leverage_cholesky_d(blas_size n, blas_size k, double* W, blas_size ldw,
                                       const double* L, blas_size ldl, double* leverage) {
    double one_d = 1.0;
    dtrsm("R", "L", "T", "N", &n, &k, &one_d, L, &ldl, W, &n);

    for(blas_size i = 0; i < n; ++i) {
        leverage[i] = ddot(&k, W + i, &n, W + i, &n);
    }
}
}

/*! For a given coefficient set beta, finds the active set by a magnitude-based rule.
 *
 *  @param[in] p The number of coefficients.
 *  @param[in] beta A pointer to the coefficients.
 *  @param[in] tolerance The tolerance to determine which values are 0.
 */
std::vector<blas_size> find_active_set(blas_size p, const double* beta, double tolerance) {
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
blas_size max_active_set_size(blas_size num_tuning, blas_size p, const double* B, blas_size ldb, double tolerance) {
    blas_size max_size = 0;

    for(blas_size i = 0; i < num_tuning; ++ i) {
		blas_size current_size = static_cast<blas_size>(std::count_if(B + ldb * i, B + ldb * i + p, [=](double x) { return std::abs(x) > tolerance; }));
        max_size = std::max(max_size, current_size);
    }

    return max_size;
}


void compute_cholesky(blas_size n, blas_size k, double* W, blas_size ldw, double* L, blas_size ldl) {
    blas_size one = 1;
    double one_d = 1.0;
    double zero_d = 0.0;
    blas_size info;

    dsyrk("L", "T", &k, &n, &one_d, W, &ldw, &zero_d, L, &ldl);
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


void lasso_compute_alo_d(blas_size n, blas_size p, blas_size m, const double* A, blas_size lda,
                         const double* B, blas_size ldb, const double* y, blas_size incy, double tolerance,
                         double* alo, double* leverage) {
    // Allocate necessary structures
    blas_size max_active = max_active_set_size(m, p, B, ldb, tolerance);

    double* L = static_cast<double*>(blas_malloc(16, max_active * max_active * sizeof(double)));
    blas_size L_active = 0;
    blas_size ldl = max_active;

    double* W = static_cast<double*>(blas_malloc(16, n * max_active * sizeof(double)));
    blas_size ldw = n;

    double* y_fitted = (double*)blas_malloc(16, n * sizeof(double));

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

        auto num_active = static_cast<blas_size>(current_index.size());

        if (num_active == 0) {
            // no active set, reset current path.
            double zero_leverage = 0.0;

            // fill the leverage to 0
            std::fill(leverage + ld_leverage * i, leverage + ld_leverage * i + n, 0.0);
            std::fill(y_fitted, y_fitted + n, 0.0);
            alo[i] = compute_alo_fitted(n, y, y_fitted, leverage + ld_leverage * i);
            L_active = 0;
            continue;
        }

        // First, we need to make sure our Cholesky decomposition is up to date.
        if (L_active > 0) {
            // update our cholesky decomposition
            lasso_update_cholesky_w_d(n, A, lda, L, ldl, W, ldw,
                static_cast<blas_size>(active_index.size()), active_index.data(),
				num_active, current_index.data());
        }
        else {
            // no existing cholesky decomposition, allocate memory and compute a new one.
            copy_active_set(n, A, lda, false, current_index, std::vector<blas_size>(), W, ldw);
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

        // compute the fitted values
        compute_fitted(n, num_active, W, B + i * ldb, 0.0, false, active_index, y_fitted);

        // compute the leverage value.
        lasso_compute_leverage_cholesky_d(n, num_active, W, ldw, L, ldl, leverage + ld_leverage * i);

        // compute the current ALO vlue.
        alo[i] = compute_alo_fitted(n, y, y_fitted, leverage + ld_leverage * i);
    }

    // free all the buffers
    blas_free(L);
    blas_free(y_fitted);
    blas_free(W);

    if (alloc_leverage) {
        blas_free(leverage);
    }
}