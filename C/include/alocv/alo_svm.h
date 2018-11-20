#ifndef WENDA_ALO_SVM_H_INCLUDED
#define WENDA_ALO_SVM_H_INCLUDED

#include "alo_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! Compute ALO for a kernel svm fit.
 *
 * @param n: The number of observations.
 * @param[in,out] K: The kernel matrix: a n x n symmetric positive definite matrix. It is modified in place for the computation.
 * @param[in] y: The vector of observed responses.
 * @param alpha: The vector of fitted dual variables.
 * @param rho: The fitted offset.
 * @param lambda: Penalization value.
 * @param tol: Tolerance for detecting support vectors.
 * @param[out, optional] leverage: If not null, a vector of length n corresponding to the computed predicted values.
 * @param[out, optional] alo_hinge: If not null, the computed ALO hinge loss.
 * @param format: The layout for the kernel matrix K.
 * 
 */
void svm_compute_alo(blas_size n, double* K, const double* y, const double* alpha,
	                 double rho, double lambda, double tol, double* alo_predicted, double* alo_hinge,
					 bool use_rfp = false);


/*! Computes RBF kernel for the given input matrix.
 *
 * @param n: The number of observations
 * @param p: The number of predictors
 * @param[in] X: The feature matrix (n x p).
 * @param gamma: The penalty.
 * @param[out] K: The computed kernel.
 * @param use_rfp: If true, K is computed in RFP format.
 * 
 */
void svm_kernel_radial(blas_size n, blas_size p, const double* X, double gamma, double* K, bool use_rfp = false);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // WENDA_ALO_SVM_H_INCLUDED