#ifndef WENDA_ALO_LASSO_H_INCLUDED
#define WENDA_ALO_LASSO_H_INCLUDED

#include "alo_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! Compute the mean square alo error along the given regularization path.
 * 
 * @param[in] n The number of observations (or rows of A).
 * @param[in] p The number of parameters (or columns of A).
 * @param[in] num_tuning The number of tuning considered (columns of B).
 * @param[in] A The regression matrix.
 * @param[in] lda The leading dimension of A.
 * @param[in] B The matrix of fitted parameters.
 * @param[in] ldb The leading dimension of B.
 * @param[in] y The observed values.
 * @param[in] incy The increment of y.
 * @param[in] tolerance The tolerance to determine the active set.
 * @param[out] alo The alo values.
 */ 
void lasso_compute_alo_d(blas_size n, blas_size p, blas_size num_tuning, double* A, blas_size lda,
                         double* B, blas_size ldb, double* y, blas_size incy, double tolerance, double* alo);


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
void lasso_update_cholesky_w_d(
	blas_size n, double* A, blas_size lda,
	double* L, blas_size ldl,
	double* W, blas_size ldw,
	blas_size len_index, blas_size* index,
	blas_size len_index_new, blas_size* index_new);

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
void lasso_compute_leverage_cholesky_d(blas_size n, blas_size k, double* W, blas_size ldw, double* L, blas_size ldl, double* leverage);

#ifdef __cplusplus
}
#endif

#endif // WENDA_ALO_LASSO_H_INCLUDED