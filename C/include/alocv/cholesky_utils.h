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

/*! Update the Cholesky representation when deleting a column.
 *
 * This function computes the update of the Cholesky decomposition L
 * when deleting a single column at location i from the original matrix.
 */
void cholesky_delete_d(blas_size n, blas_size i, double* L, blas_size ldl, double* Lo, blas_size ldol);

/*! Update the Cholesky representation when adding o column.
 *
 * This function computes the update of the Cholesky decomposition L
 * when adding a single column at the end of the original matrix.
 */
void cholesky_append_d(blas_size n, double* L, blas_size ldl, double* b, blas_size incb, double c, double* Lo, blas_size ldol);

#ifdef __cplusplus
}
#endif

#endif