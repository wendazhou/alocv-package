#include "alocv/alo_lasso.h"
#include "alocv/cholesky_utils.h"
#include "blas_configuration.h"

#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>
#include <iostream>
#include <cmath>


void lasso_update_cholesky_d(int n, double* A, int lda, double* L, int ldl, double* Lo, int ldol,
                           int len_index, int* index, int len_index_new, int* index_new) {
    // First compute the columns to add and remove from the matrix to update it.
    std::vector<int> start_index(index, index + len_index);
    std::vector<int> end_index(index_new, index_new + len_index_new);

    std::vector<int> active_index(start_index);

    std::sort(start_index.begin(), start_index.end());
    std::sort(end_index.begin(), end_index.end());

    std::vector<int> index_added;
    std::vector<int> index_removed;

    std::set_difference(
        end_index.begin(), end_index.end(),
        start_index.begin(), start_index.end(),
        std::back_inserter(index_added));
    
    std::set_difference(
        start_index.begin(), start_index.end(),
        end_index.begin(), end_index.end(),
        std::back_inserter(index_removed));
    
    // Delete all unnecessary columns first.
    for(int i: index_removed) {
        auto it = std::find(active_index.begin(), active_index.end(), i);
        int loc = std::distance(active_index.begin(), it);

        cholesky_delete_d(active_index.size(), loc, L, ldl, Lo, ldol);

        active_index.erase(it);
        L = Lo;
        ldl = ldol;
    }

    double* b = static_cast<double*>(blas_malloc(16, end_index.size() * sizeof(double)));

    // Append all necessary indices to reach the desired state.
    for(int i: index_added) {
        int one = 1;
        double* current_col = A + lda * i;
        double c = ddot(&n, current_col, &one, current_col, &one);

        for(int j = 0; j < active_index.size(); ++j) {
            b[j] = ddot(&n, A + active_index[j] * lda, &one, A + i * lda, &one);
        }

        cholesky_append_d(active_index.size(), L, ldl, b, 1, c, Lo, ldol);

        active_index.push_back(i);
        L = Lo;
        ldl = ldol;
    }

    blas_free(b);

    // index_new contains the corresponding set of indices.
    assert(active_index.size() == len_index_new);
    std::copy(active_index.begin(), active_index.end(), index_new);
}


void lasso_compute_leverage_cholesky_d(int n, double* A, int lda, double* L, int ldl, int k, int* index, double* leverage) {
    double* W = static_cast<double*>(blas_malloc(16, n * k * sizeof(double)));

    for(int i = 0; i < k; ++i) {
        memcpy(W + n * i, A + index[i] * lda, n * sizeof(double));
    }

    double one_d = 1.0;
    dtrsm("R", "L", "T", "N", &n, &k, &one_d, L, &ldl, W, &n);

    for(int i = 0; i < n; ++i) {
        leverage[i] = ddot(&k, W + i, &n, W + i, &n);
    }

    blas_free(W);
}

std::vector<int> find_active_set(int p, double* beta, double tolerance) {
    std::vector<int> result;

    for(int i = 0; i < p; ++i) {
        if (std::abs(beta[i]) > tolerance) {
            result.push_back(i);
        }
    }

    return result;
}

double compute_alo(int n, int p, double* A, int lda, double* y, double* beta, double* leverage, int incl) {
    double* temp = static_cast<double*>(blas_malloc(16, n * sizeof(double)));

    // temp = y
    memcpy(temp, y, n * sizeof(double));

    double one_d = 1.0;
    double min_one_d = -1.0;
    int one_i = 1;

    // temp = X * beta - y
    dgemv("N", &n, &p, &one_d, A, &lda, beta, &one_i, &min_one_d, temp, &one_i);

    double acc = 0;

    for(int i = 0; i < n; ++i) {
        double res = temp[i] / (1 - leverage[incl * i]);
        acc += res * res;
    }

    blas_free(temp);

    return acc;
}

void compute_cholesky(int n, int k, int* index, double* A, int lda, double* L, int ldl) {
    int one = 1;

    for(int j = 0; j < k; ++j) {
        for(int i = j; i < k; ++i) {
            L[i + ldl * j] = ddot(&n, A + lda * index[i], &one, A + lda * index[j], &one);
        }
    }

    int info;

    dpotrf("L", &k, L, &ldl, &info);
}

void lasso_compute_alo_d(int n, int p, int m, double* A, int lda, double* B, int ldb, double* y, int incy, double tolerance, double* alo) {
    std::vector<int> active_index;
    double* current_cholesky = nullptr;
    double* leverage = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
    int current_cholesky_size = 0;

    for(int i = 0; i < m; ++i) {
        std::vector<int> current_index = find_active_set(p, B + ldb * i, tolerance);

        auto num_active = current_index.size();

        if (num_active == 0) {
            // no active set, reset current path.
            double zero_leverage = 0.0;
            alo[i] = compute_alo(n, p, A, lda, y, B + ldb * i, &zero_leverage, 0);
            blas_free(current_cholesky);
            current_cholesky = 0;
            continue;
        }

        // First, we need to make sure our Cholesky decomposition is up to date.
        if (current_cholesky) {
            // update our cholesky decomposition
            double* new_cholesky;
            int new_cholesky_size;

            if (current_cholesky_size < num_active) {
                // need to allocate space for new cholesky decomposition.
                new_cholesky_size = num_active;
                new_cholesky = static_cast<double*>(blas_malloc(16, new_cholesky_size * new_cholesky_size * sizeof(double)));
            }
            else {
                // can reuse existing allocation.
                new_cholesky = current_cholesky;
                new_cholesky_size = current_cholesky_size;
            }

            lasso_update_cholesky_d(n, A, lda, current_cholesky, current_cholesky_size,
                                    new_cholesky, new_cholesky_size,
                                    active_index.size(), active_index.data(),
                                    current_index.size(), current_index.data());
            
            if (new_cholesky != current_cholesky) {
                // if we had to allocate, free previous memory and reset pointers.
                blas_free(current_cholesky);
                current_cholesky = new_cholesky;
                current_cholesky_size = new_cholesky_size;
            }
        }
        else {
            // no existing cholesky decomposition, allocate memory and compute a new one.
            current_cholesky_size = num_active;
            current_cholesky = static_cast<double*>(blas_malloc(16, current_cholesky_size * current_cholesky_size * sizeof(double)));
            compute_cholesky(n, current_cholesky_size, current_index.data(), A, lda, current_cholesky, current_cholesky_size);
        }

        // update the active index
        std::swap(active_index, current_index);

        // Now that we have the Cholesky decomposition, let's compute the leverage values.
        if (num_active >= n) {
            // special case where the active set is of the same size as the
            // number of observations. The ALO estimate of risk is infinite.
            alo[i] = INFINITY;
            continue;
        }

        // compute the leverage value.
        lasso_compute_leverage_cholesky_d(n, A, lda, current_cholesky, current_cholesky_size,
                                          num_active, active_index.data(), leverage);
        // compute the current ALO vlue.
        alo[i] = compute_alo(n, p, A, lda, y, B + ldb * i, leverage, 1);
    }

    // free all the buffers
    mkl_free(current_cholesky);
    mkl_free(leverage);
}