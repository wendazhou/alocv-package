#ifndef WENDA_CHOLESKY_UTILS_H_INCLUDED
#define WENDA_CHOLESKY_UTILS_H_INCLUDED

#include "alo_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! Update the Cholesky representation of a given matrix.
 *
 *  This function computes the update of the cholesky decomposition L
 *  by a rank one perturbation given by x.
 */
void cholesky_update_d(blas_size n, double* L, blas_size ldl, double* x, blas_size incx);

/*! Downdate the Cholesky representation of a given matrix.
 *
 * This function computes the downdate of the Cholesky decomposition L
 * by a rank one perturbation given by x. No checking is done to ensure validity.
 * Note that this function mutates the variable x to store intermediate results.
 *
 * @param[in] n The size of the current decomposition.
 * @param[in,out] L The decomposition to update.
 * @param[in] ldl The leading dimension of L. Must be at least n.
 * @param[in,out] x The vector to downdate L by.
 * @param[in] incx The increment of x.
 */
void cholesky_downdate_d(blas_size n, double* L, blas_size ldl, double* x, blas_size incx);

/*! Update the Cholesky representation when deleting a column.
 *
 * This function computes the update of the Cholesky decomposition L
 * when deleting a single column at location i from the original matrix.
 */
void cholesky_delete_d(blas_size n, blas_size i, double* L, blas_size ldl, double* Lo, blas_size ldlo);


/*! Update the Cholesky representation when deleting a column.
 *
 * This function computes the update of the Cholesky decomposition L
 * whene deleting a single column at location i from the original matrix.
 * The matrix L is mutated in place to reflect the result.
 *
 * @param[in] n The size of the current Cholesky decomposition
 * @param[in] i The index of the column to delete.
 * @param[in,out] L The current decomposition.
 * @param[in] ldl The leading dimension of L.
 */
void cholesky_delete_inplace_d(blas_size n, blas_size i, double* L, blas_size ldl);

/*! Update the Cholesky representation when adding o column.
 *
 * This function computes the update of the Cholesky decomposition L
 * when adding a single column at the end of the original matrix.
 */
void cholesky_append_d(blas_size n, double* L, blas_size ldl, double* b, blas_size incb, double c, double* Lo, blas_size ldlo);


/*! Update the Cholesky representation when adding one column inplace.
 *
 * This function computes the update of the Cholesky decomposition L,
 * when adding a single row at the end of the original matrix.
 *
 * This function operates entirely inplace, and assumes that the row being added
 * to the matrix is stored in the last row of L. This function only accesses and modifies
 * the lower triangular part of L.
 *
 * @param[in] n The size of the current Cholesky decomposition stored in L.
 * @param[in,out] L The current decomposition, along with the new row added to the matrix.
 *					Must be of size at least (n + 1) * ldl
 * @param[in] ldl The leading dimension of L, must be at least n + 1.
 */
void cholesky_append_inplace_d(blas_size n, double* L, blas_size ldl);

#ifdef __cplusplus
}
#endif

#endif