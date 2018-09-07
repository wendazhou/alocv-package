#include "alocv/alo_config.h"
#include <vector>

/*! For a given coefficient set beta, finds the active set by a magnitude-based rule.
 *
 *  @param[in] p The number of coefficients.
 *  @param[in] beta A pointer to the coefficients.
 *  @param[in] tolerance The tolerance to determine which values are 0.
 */
std::vector<blas_size> find_active_set(blas_size p, const double* beta, double tolerance);

/*! Finds the largest active set in the given set of solutions.
 *
 * @param num_tuning The number of rows of B
 * @param p The number of columns of B
 * @param[in] B A pointer to the coefficients for each value of the regularizer.
 * @param ldb the leading dimension of B.
 * @param tolerance The tolerance to determine which values are 0.
 */
blas_size max_active_set_size(blas_size num_tuning, blas_size p, const double* B, blas_size ldb, double tolerance);


/*! Computes the ALO value given the fitted observations and leverage.
 *
 * @param n: The number of observations
 * @param[in] y: A vector containing the true observed value.
 * @param[in] y_fitted: A vector containing the fitted value (in link space).
 * @param[in] leverage: A vector containing the leverage for each observation.
 * 
 * @returns The ALO mean-squared error.
 */
double compute_alo_fitted(blas_size n, const double* y, const double* y_fitted, const double* leverage);


/*! Copies over the active predictors to a packed matrix.
 *
 * @param n The number of observations
 * @param[in] A The matrix of all predictors.
 * @param lda The leading dimension of A.
 * @param has_intercept Whether to add an intercept column.
 * @param index_active The index of currently active predictors.
 * @param index_added Another optional vector of indices to append to the active predictors.
 * @param[out] W A matrix to copy the predictors to.
 * @param ldw The leading dimension of W, must be at least n.
 * 
 */
void copy_active_set(blas_size n, const double* A, blas_size lda, bool has_intercept,
                   std::vector<blas_size> const& index_active, std::vector<blas_size> const& index_added,
                   double* W, blas_size ldw);


/*! Compute fitted response in link space.
 *
 * @param n The number of observations
 * @param k The number of non-zero parameters.
 * @param XE[in] A matrix containing the active set predictors
 * @param beta[in] A vector containing the parameters.
 * @param a0 A double representing the value of the intercept. Ignored if no intercept.
 * @param has_intercept If true, signifies that an intercept was fitted.
 * @param index A list containing the indices of the active set.
 * @param y_fitted[out] A vector of length n which will be filled with the fitted link space result.
 * 
 */
void compute_fitted(blas_size n, blas_size k, const double* XE,
                    const double* beta, double a0, bool has_intercept,
                    const std::vector<blas_size>& index, double* y_fitted);