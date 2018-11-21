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
 * @param[out,optional] leverage If not NULL, a matrix which stores the leverage values for each observation
                        and tuning.
 */ 
void lasso_compute_alo_d(blas_size n, blas_size p, blas_size num_tuning, const double* A, blas_size lda,
                         const double* B, blas_size ldb, const double* y, blas_size incy, double tolerance,
                         double* alo, double* leverage);


#ifdef __cplusplus
}
#endif

#endif // WENDA_ALO_LASSO_H_INCLUDED