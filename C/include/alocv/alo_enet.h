#ifndef WENDA_ALO_ENET_H_INCLUDED
#define WENDA_ALO_ENET_H_INCLUDED

#include "alo_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! Computes the ALOCV value for the elastic-net problem.
 *
 * @param n: The number of observations.
 * @param p: The number of parameters.
 * @param m: The number of fitted values.
 * @param[in] A: The predictor matrix, a n x p matrix in column-major order.
 * @param[in] lda: The leading dimension of A.
 * @param[in] B: The fitted values, a p x m matrix in column-major order.
 * @param ldb: The leading dimension of B.
 * @param[in] y: The vector of observed responses, a vector of length n.
 * @param[in] lambda: The vector of regularization values, a vector of length m.
 * @param alpha: The elastic net parameter.
 * @param[out] alo: A vector of length m, will contain the ALOCV value for each tuning.
 * @param[out, optional] leverage: If provided, a n x m matrix which will contain the
 *      leverage values for each observation and tuning.
 * 
 */
void enet_compute_alo_d(blas_size n, blas_size p, blas_size m, const double* A, blas_size lda,
                        const double* B, blas_size ldb, const double* y, const double* lambda, double alpha,
                        bool has_intercept, double tolerance,
                        double* alo, double* leverage = 0);

#ifdef __cplusplus
} // extern "C"
#endif


#endif // WENDA_ALO_ENET_H_INCLUDED