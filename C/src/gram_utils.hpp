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


enum class MatrixTranspose {
	Identity,
	Transpose
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


/*! Computes a matrix-matrix product where the left input matrix is triangular.
 *
 * @param transa: Whether to transpose the matrix A
 * @param m: The number of rows and columns of A
 * @param n: The number of columns of B.
 * @param A[in]: The triangular matrix A, either in full or RFP format.
 * @param B[in, out]: The RHS of the operation, and where the result is stored.
 * @param ldb: The leading dimension of B.
 * @param format: The format of A.
 *
 */
void triangular_multiply(MatrixTranspose transa, blas_size m, blas_size n, const double* A, double* B, blas_size ldb, SymmetricFormat format);


/*! Adds the given value to the diagonal of the matrix (represented in the specified format).
 *
 * If specified, this function will not increment the top-left element of the matrix.
 * 
 * @param p The size of the matrix.
 * @param[in, out] L The matrix in the given format.
 * @param value The value to add to the diagonal.
 * @param skip_first If true, the value is not added to the first element (top left element).
 * 
 */
void offset_diagonal(blas_size p, double* L, double value, bool skip_first, SymmetricFormat format);


inline double diagonal_element(blas_size p, double* L, blas_size index, SymmetricFormat format) {
    if(format == SymmetricFormat::Full) {
        return L[index + p * index];
    } else {
        if(p % 2 == 0) {
            blas_size row_offset = 1;
			blas_size ldl = p + 1;

            if(index >= p / 2) {
                index -= p / 2;
                row_offset = 0;
            }

            return L[index + ldl * index + row_offset];
        } else {
            blas_size col_offset = 0;
			blas_size ldl = p;

            if(index >= (p + 1) / 2) {
                index -= (p + 1) / 2;
                col_offset = 1;
            }

            return L[index + ldl * (index + col_offset)];
        }
    }
}

#endif // WENDA_GRAM_UTILS_H_INCLUDED