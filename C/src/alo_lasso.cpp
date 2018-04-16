#include "alocv/alo_lasso.h"
#include "alocv/cholesky_utils.h"
#include "blas_configuration.h"

#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>
#include <iostream>


void lasso_update_cholesky_d(int n, double* A, int lda, double* L, int ldl, double* Lo, int lodl,
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

        cholesky_delete_d(active_index.size(), loc, L, ldl, Lo, lodl);

        active_index.erase(it);
        L = Lo;
        ldl = lodl;
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

        cholesky_append_d(active_index.size(), L, ldl, b, 1, c, Lo, lodl);

        active_index.push_back(i);
        L = Lo;
        ldl = lodl;
    }

    blas_free(b);

    // index_new contains the corresponding set of indices.
    assert(active_index.size() == len_index_new);
    std::copy(active_index.begin(), active_index.end(), index_new);
}