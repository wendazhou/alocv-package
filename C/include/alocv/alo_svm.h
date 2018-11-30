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
 * @param use_rfp: If true, indicates that K is communicated in RFP format.
 * @param use_pivoting: If true, indicates that pivoting should be used (enables computation on potentially singular kernels).
 *                      Note that this is not compatible with use_rfp, and will be ignored if use_rfp is set to true.
 * 
 */
void svm_compute_alo(blas_size n, double* K, const double* y, const double* alpha,
	                 double rho, double lambda, double tol, double* alo_predicted, double* alo_hinge,
					 bool use_rfp = false, bool use_pivoting = false);


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

/*! Computes polynomial kernel for the given input matrix.
 *
 * @param n: The number of observations.
 * @param p: The number of predictors.
 * @param[in] X: the feature matrix (n x p).
 * @param[out] K: The computed kernel.
 * @param gamma: Coefficient for penalty.
 * @param degree: The degree of the polynomial.
 * @param coef0: The offset of the polynomial.
 * @param use_rfp: If true, K is computed in RFP format.
 *
 */
void svm_kernel_polynomial(blas_size n, blas_size p, const double* X, double* K, double gamma, double degree, double coef0, bool use_rfp = false);

/*! Computes linear kernel for the given input matrix.
 *
 * @param n: The number of observations
 * @param p: The number of predictors.
 * @param[in] X: The feature matrix (n x p).
 * @param[out] K: The computed kernel.
 * @param use_rfp: If true, K is computed in RFP format.
 *
 */
void svm_kernel_linear(blas_size n, blas_size p, const double* X, double* K, bool use_rfp = false);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // WENDA_ALO_SVM_H_INCLUDED