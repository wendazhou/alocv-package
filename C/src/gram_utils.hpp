#ifndef WENDA_GRAM_UTILS_H_INCLUDED
#define WENDA_GRAM_UTILS_H_INCLUDED

#include <alocv/alo_config.h>

/** Describes the format used to store symmetric and triangular matrices
 * when required. 
 * 
 */
enum class SymmetricFormat {
    Full,
    RFP
};

/*! Computes the Gram matrix of the given dataset.
 *
 * @param n The number of rows of XE
 * @param p The number of columns of XE
 * @param XE[in] a n x p matrix representing the data
 * @param lde The leading dimension of XE
 * @param L[out] A symmetric matrix which will contain the inner product of columns of XE.
 * 
 */
void compute_gram(blas_size n, blas_size p, const double* XE, blas_size lde, double* L, SymmetricFormat format);

/*! Computes the Cholesky decomposition of the matrix.
 *
 * @param p The number of rows and columns of L
 * @param L[in, out] The matrix for which to compute the Cholesky decomposition.
 * 
 */
int compute_cholesky(blas_size p, double* L, SymmetricFormat format);

/*! Solves the triangular inverse problem for the given matrix.
 *
 * @param n The number of rows of XE
 * @param p The number of columns of XE
 * @param L[in] A lower-triangular matrix of size p x p.
 * @param XE[in, out] The matrix for which to compute the solution.
 * @param lde The leading dimension of XE.
 * 
 */
void solve_triangular(blas_size n, blas_size p, const double* L, double* XE, blas_size lde, SymmetricFormat format);

#endif // WENDA_GRAM_UTILS_H_INCLUDED