#ifndef WENDA_ALO_SVM_H_INCLUDED
#define WENDA_ALO_SVM_H_INCLUDED

#include "alo_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! Compute ALO for a kernel svm fit.
 *
 * @param n: The number of observations.
 * @param[in] K: The kernel matrix: a n x n symmetric positive definite matrix.
 * @param ldk: The leading dimension of K, must be at least n.
 * @param[in] y: The vector of observed responses.
 * @param alpha: The vector of fitted dual variables.
 * @param rho: The fitted offset.
 * @param lambda: Penalization value.
 * @param tol: Tolerance for detecting support vectors.
 * @param[out, optional] leverage: If not null, a vector of length n corresponding to the computed predicted values.
 * @param[out, optional] alo_hinge: If not null, the computed ALO hinge loss.
 * 
 */
void svm_compute_alo(blas_size n, const double* K, blas_size ldk, const double* y, const double* alpha,
                     double rho, double lambda, double tol, double* alo_predicted, double* alo_hinge);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // WENDA_ALO_SVM_H_INCLUDED