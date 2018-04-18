#include "alocv/cholesky_utils.h"
#include "blas_configuration.h"
#include "math.h"
#include "assert.h"
#include "string.h"
#include "stdio.h"


void cholesky_update_d(blas_size n, double* L, blas_size ldl, double* x, blas_size incx) {
    double c, s;
    blas_size incr = 1;

    for(int k = 0; k < n; k++) {
        blas_size m = n - k - 1;
        drotg(L + ldl * k + k, x + incx * k, &c, &s);
        drot(&m, L + ldl * k + k + 1, &incr, x + incx * (k + 1), &incx, &c, &s);
    }
}

/* Computes a rank-1 Cholesky downdate. This function implements the algorithm
 * described in "A modification to the LINPACK downdating algorithm" DOI:10.1007/BF01933218.
 */
void cholesky_downdate_d(blas_size n, double* L, blas_size ldl, double* x, blas_size incx) {
	blas_size one_i = 1;

	double alpha;
	double alpha_prev = 1.0;
	double beta;
	double beta_prev = 1.0;

	for(blas_size k = 0; k < n; k++) {
		blas_size remaining = n - k - 1;

		double a = -x[k * incx] / L[k * ldl + k];
		alpha = alpha_prev - a * a;
		beta = sqrt(alpha);

		daxpy(&remaining, &a, L + k * ldl + k + 1, &one_i, x + (k + 1) * incx, &one_i);

		blas_size n_minus_k = n - k;
		double beta_scale = beta / beta_prev;
		dscal(&n_minus_k, &beta_scale, L + ldl * k + k, &one_i);

		double a_beta_scale = a / (beta * beta_prev);
		daxpy(&remaining, &a_beta_scale, x + (k + 1) * incx, &incx, L + k * ldl + k + 1, &one_i);

		alpha_prev = alpha;
		beta_prev = beta;
	}
}

void cholesky_delete_d(blas_size n, blas_size i, double* L, blas_size ldl, double* Lo, blas_size ldlo) {
    blas_size s22_length = n - i - 1;
    blas_size one = 1;

    double* temp = blas_malloc(16, s22_length * sizeof(double));
    dcopy(&s22_length, L + i * ldl + (i + 1), &one, temp, &one);

	if (L != Lo || ldl != ldlo) {
		dlacpy("L", &i, &i, L, &ldl, Lo, &ldlo);
	}
    dlacpy("A", &s22_length, &i, L + i + 1, &ldl, Lo + i, &ldlo);
    dlacpy("L", &s22_length, &s22_length, L + (i + 1) * ldl + (i + 1), &ldl, Lo + i * ldlo + i, &ldlo);

    cholesky_update_d(s22_length, Lo + i * ldlo + i, ldlo, temp, 1);
    blas_free(temp);
}

void cholesky_delete_inplace_d(blas_size n, blas_size i, double* L, blas_size ldl) {
    blas_size s22_length = n - i - 1;
    blas_size one = 1;

    double* temp = blas_malloc(16, s22_length * sizeof(double));
	memcpy(temp, L + i * ldl + (i + 1), s22_length * sizeof(double));
    dlacpy("A", &s22_length, &i, L + i + 1, &ldl, L + i, &ldl);
    dlacpy("L", &s22_length, &s22_length, L + (i + 1) * ldl + (i + 1), &ldl, L + i * ldl + i, &ldl);

    cholesky_update_d(s22_length, L + i * ldl + i, ldl, temp, 1);
	blas_free(temp);
}

void cholesky_append_d(blas_size n, double* L, blas_size ldl, double* b, blas_size incb, double c, double* Lo, blas_size ldlo) {
	dlacpy("L", &n, &n, L, &ldl, Lo, &ldlo);
	dcopy(&n, b, &incb, Lo + n, &ldlo);
	Lo[n * ldlo + n] = c;

	cholesky_append_inplace_d(n, Lo, ldlo);
}

void cholesky_append_inplace_d(blas_size n, double* L, blas_size ldl) {
	assert(ldl >= n + 1);

	blas_size one = 1;
	double one_d = 1.0;

	dtrsm("R", "L", "C", "N", &one, &n, &one_d, L, &ldl, L + n, &ldl);
    double border_inner = ddot(&n, L + n, &ldl, L + n, &ldl);
	L[n * ldl + n] = sqrt(L[n * ldl + n] - border_inner);
}

void cholesky_append_inplace_multiple_d(blas_size n, blas_size k, double* L, blas_size ldl) {
	// Appending to the end of the Cholesky decomposition can be viewed in 3 parts
	// 1) the existing upper left triangle is preserved
	// 2) the bottom left rectangular part beneath that should be inverted by the existing Cholesky.
	// 3) we need to compute a new Cholesky for the remaining bottom right triangle.

	assert(ldl >= n + k);

	double one_d = 1.0;
	double minus_one_d = -1.0;

	dtrsm("R", "L", "T", "N", &k, &n, &one_d, L, &ldl, L + n, &ldl);
#ifdef USE_MKL
	dgemmt("L", "N", "T", &k, &n, &minus_one_d, L + n, &ldl, L + n, &ldl, &one_d, L + ldl * n + n, &ldl);
#else
	dgemm("N", "T", &k, &k, &n, &minus_one_d, L + n, &ldl, L + n, &ldl, &one_d, L + ldl * n + n, &ldl);
#endif

	blas_size info;
	dpotrf("L", &k, L + ldl * n + n, &ldl, &info);
}
