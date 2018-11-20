#include "gram_utils.hpp"
#include "blas_configuration.h"


void compute_gram(blas_size n, blas_size p, const double* XE, blas_size lde, double* L, SymmetricFormat format) {
    const double one = 1;
    const double zero = 0;

    if(format == SymmetricFormat::Full) {
        dsyrk("L", "T", &p, &n, &one, XE, &lde, &zero, L, &p);
        return;
    }

#ifndef ALOCV_LAPACK_NO_RFP
    dsfrk("N", "L", "T", &p, &n, &one, XE, &lde, &zero, L);
#else
    const bool is_odd = p % 2;
    const blas_size p2 = p / 2;
    const blas_size p1 = p - p2;
    const blas_size ldl = is_odd ? p : p + 1;

    dsyrk("L", "T", &p1, &n, &one, XE, &lde, &zero, L + (is_odd ? 0 : 1), &ldl);
    dsyrk("U", "T", &p2, &n, &one, XE + p1 * lde, &lde, &zero, L + (is_odd ? p : 0), &ldl);
    dgemm("T", "N", &p2, &p1, &n, &one, XE + p1 * lde, &lde, XE, &lde, &zero, L + p1 + (is_odd ? 0 : 1), &ldl);
#endif
}

/*! Computes the Cholesky decomposition of the matrix in RFP format. */
int compute_cholesky(blas_size p, double* L, SymmetricFormat format) {
    int info;

    if(format == SymmetricFormat::Full) {
        dpotrf("L", &p, L, &p, &info);
        return info;
    }

#ifndef ALOCV_LAPACK_NO_RFP
    dpftrf("N", "L", &p, L, &info);
#else
    const bool is_odd = p % 2;
    const blas_size p2 = p / 2;
    const blas_size p1 = p - p2;

    const blas_size ldl = is_odd ? p : p + 1;

    const blas_size o1 = is_odd ? 0 : 1;
    const blas_size o2 = is_odd ? p1 : p1 + 1;
    const blas_size o3 = is_odd ? p : 0;

    dpotrf("L", &p1, L + o1, &ldl, &info);

    if(info) return info;

    const double one = 1;
    const double neg_one = -1;

    dtrsm("R", "L", "T", "N", &p2, &p1, &one, L + o1, &ldl, L + o2, &ldl);
    dsyrk("U", "N", &p2, &p1, &neg_one, L + o2, &ldl, &one, L + o3, &ldl);
    dpotrf("U", &p2, L + o3, &ldl, &info);
#endif
    return info;
}

void solve_triangular(blas_size n, blas_size p, const double* L, double* XE, blas_size lde, SymmetricFormat format) {
    const double one = 1;

    if(format == SymmetricFormat::Full) {
        dtrsm("R", "L", "T", "N", &n, &p, &one, L, &p, XE, &lde);
        return;
    }

#ifndef ALOCV_LAPACK_NO_RFP
    dtfsm("N", "R", "L", "T", "N", &n, &p, &one, L, XE, &lde);
#else
    const bool is_odd = p % 2;
    const blas_size p2 = p / 2;
    const blas_size p1 = p - p2;

    const blas_size ldl = is_odd ? p : p + 1;

    const double neg_one = -1;

    dtrsm("R", "L", "T", "N", &n, &p1, &one, L + (is_odd ? 0 : 1), &ldl, XE, &lde);
    dgemm("N", "T", &n, &p2, &p1, &neg_one, XE, &lde, L + p1 + (is_odd ? 0 : 1), &ldl, &one, XE + p1 * lde, &lde);
    dtrsm("R", "U", "N", "N", &n, &p2, &one, L + (is_odd ? p : 0), &ldl, XE + p1 * lde, &lde);
#endif
}


void triangular_multiply(MatrixTranspose transa, blas_size m, blas_size n, const double* A, double* B, blas_size ldb, SymmetricFormat format) {
	const double one = 1.0;

	if (format == SymmetricFormat::Full) {
		dtrmm("L", "L", transa == MatrixTranspose::Identity ? "N" : "T", "N", &m, &n, &one, A, &m, B, &ldb);
		return;
	}

	const bool is_odd = m % 2 == 1;
	const blas_size ldar = is_odd ? m : m + 1;
	const blas_size m1 = (m + 1) / 2;
	const blas_size m2 = m - m1;

	const double* L22 = A + (is_odd ? ldar : 0);
	const double* L11 = A + (is_odd ? 0 : 1);
	const double* L12 = A + m1 + (is_odd ? 0 : 1);

	if (transa == MatrixTranspose::Identity) {
		dtrmm("L", "U", "T", "N", &m2, &n, &one, L22, &ldar, B + m1, &ldb);
		dgemm("N", "N", &m2, &n, &m1, &one, L12, &ldar, B, &ldb, &one, B + m1, &ldb);
		dtrmm("L", "L", "N", "N", &m1, &n, &one, L11, &ldar, B, &ldb);
	}
	else {
		dtrmm("L", "L", "T", "N", &m1, &n, &one, L11, &ldar, B, &ldb);
		dgemm("T", "N", &m1, &n, &m2, &one, L12, &ldar, B + m1, &ldb, &one, B, &ldb);
		dtrmm("L", "U", "N", "N", &m2, &n, &one, L22, &ldar, B + m1, &ldb);
	}
}


void symmetric_multiply(blas_size m, blas_size n, const double* A, double* B, blas_size ldb, SymmetricFormat format) {
	const double one = 1.0;
	const double zero = 0.0;

	if (format == SymmetricFormat::Full) {
		dsymm("L", "L", &m, &n, &one, A, &m, B, &ldb, &zero, B, &ldb);
		return;
	}

	const bool is_odd = m % 2 == 1;
	const blas_size ldar = is_odd ? m : m + 1;
	blas_size m1 = (m + 1) / 2;
	blas_size m2 = m - m1;

	const double* L22 = A + (is_odd ? ldar : 0);
	const double* L11 = A + (is_odd ? 0 : 1);
	const double* L12 = A + m1 + (is_odd ? 0 : 1);

	// due to a bug? in Intel MKL, we have to allocate a copy
	auto temp_storage = blas_unique_alloc<double>(16, m * n);
	double* temp = temp_storage.get();

	dlacpy("N", &m, &n, B, &ldb, temp, &m);

	dsymm("L", "U", &m2, &n, &one, L22, &ldar, temp + m1, &m, &zero, B + m1, &ldb);
	dgemm("N", "N", &m2, &n, &m1, &one, L12, &ldar, B, &ldb, &one, B + m1, &ldb);

	dsymm("L", "L", &m1, &n, &one, L11, &ldar, temp, &m, &zero, B, &ldb);
	dgemm("T", "N", &m1, &n, &m2, &one, L12, &ldar, temp + m1, &m, &one, B, &ldb);
}

void offset_diagonal(blas_size p, double* L, double value, bool skip_first, SymmetricFormat format) {
    if(format == SymmetricFormat::Full) {
        for(int i = skip_first; i < p; ++i) {
            L[i * p + i] += value;
        }

        return;
    } else {
        if(p % 2 == 1) {
            // odd case

            blas_size ldl = p;

            if(!skip_first) {
                L[0] += value;
            }

            for(blas_size i = 1; i < (p + 1) / 2; ++i) {
                L[i + ldl * i] += value;
                L[i + ldl * i - 1] += value;
            }
        } else {
            // even case

            blas_size ldl = p + 1;

            L[0] += value;

            if(!skip_first) {
                L[1] += value;
            }

            for(blas_size i = 1; i < p / 2; ++i) {
                L[i + ldl * i] += value;
                L[i + ldl * i + 1] += value;
            }
        }
    }
}
