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


/*! Computes a matrix-matrix product where the left input matrix is symmetric.
 *
 * @param m: The number of rows and columns of A
 * @param n: The number of columns of B.
 * @param A[in]: The symmetric matrix A, either in full (lower triangular) or RFP format.
 * @param B[in, out]: The RHS of the operation, and where the result is stored.
 * @param ldb: The leading dimension of B.
 * @param format: The format of A.
 *
 */
void symmetric_multiply(blas_size m, blas_size n, const double* A, double* B, blas_size ldb, SymmetricFormat format);


/*! Copies a column from a given symmetric or triangular matrix in RFP format to the destination pointer.
 *
 * @param n: The size of the matrix A.
 * @param A: The matrix to copy from.
 * @param k: The index of the column to copy.
 * @param B: The array to copy the column of A to.
 * @param format: The format of A.
 * @param copy_symmetric: If true, indicates that we wish to extend the copied column of A assuming
 *						  that A is a symmetric matrix.
 *
 */
inline void copy_column(blas_size n, const double* A, blas_size k, double* B, MatrixTranspose transa,
					    SymmetricFormat format, bool copy_symmetric = false);


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


/*!  Gets the given diagonal element from a matrix in the RFP format.
 * @param p: The size of the matrix.
 * @param L: The matrix in the given format.
 * @param index: The index of the diagonal element to obtain.
 * @param format: The format of the given matrix.
 *
 */
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

template<typename ItS, typename ItD>
void strided_copy(ItS first, ItS end, ItD dest, blas_size stride_in, blas_size stride_out) {
	while (first < end) {
		*dest = *first;
		first += stride_in;
		dest += stride_out;
	}
}

#include <algorithm>

inline void copy_column(blas_size n, const double* A, blas_size k, double* B, MatrixTranspose transa,
					    SymmetricFormat format, bool copy_symmetric) {
	if (copy_symmetric) {
		// when copying with symmetric extension, this is equivalent to copying the triangular array
		// from the transpose and the non-transposed.
		copy_column(n, A, k, B, transa == MatrixTranspose::Identity ? MatrixTranspose::Transpose : MatrixTranspose::Identity,
			format, false);
	}

	if (format == SymmetricFormat::Full) {
		if (transa == MatrixTranspose::Identity) {
			std::copy(A + k * n + k, A + k * n + n, B + k);
		}
		else {
			strided_copy(A + k, A + k * n + k + 1, B, n, 1);
		}

		return;
	}

	bool is_odd = n % 2 == 1;
	blas_size n1 = (n + 1) / 2;
	blas_size n2 = n - n1;
	blas_size ldar = is_odd ? n : n + 1;

	if (transa == MatrixTranspose::Identity) {
		if (k < n1) {
			blas_size offset = is_odd ? 0 : 1;
			std::copy(A + k * ldar + k + offset, A + k * ldar + n + offset, B + k);
		}
		else {
			blas_size offset = is_odd ? 1 : 0;
			strided_copy(A + (k - n1 + offset) * ldar + (k - n1), A + n1 * ldar + (k - n1), B + k, ldar, 1);
		}
	}
	else {

		{
			blas_size offset = is_odd ? 0 : 1;
			strided_copy(A + k + offset, A + k + offset + std::min(k + 1, n1) * ldar, B, ldar, 1);
		}

		if (k >= n1) {
			// copy stragglers
			blas_size offset = is_odd ? 1 : 0;
			auto first = A + (k - n1 + offset) * ldar;
			std::copy(first, first + (k - n1 + 1), B + n1);
		}
	}
}

#endif // WENDA_GRAM_UTILS_H_INCLUDED