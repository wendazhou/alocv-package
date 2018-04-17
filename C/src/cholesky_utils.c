#include "alocv/cholesky_utils.h"
#include "blas_configuration.h"
#include "math.h"
#include "assert.h"


void cholesky_update_d(blas_size n, double* L, blas_size ldl, double* x, blas_size incx) {
    double c, s;
    int incr = 1;

    for(int k = 0; k < n; k++) {
        int m = n - k - 1;
        drotg(L + ldl * k + k, x + incx * k, &c, &s);
        drot(&m, L + ldl * k + k + 1, &incr, x + incx * (k + 1), &incx, &c, &s);
    }
}

void cholesky_delete_d(blas_size n, blas_size i, double* L, blas_size ldl, double* Lo, blas_size ldlo) {
    int s22_length = n - i - 1;
    int one = 1;

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

void cholesky_append_d(blas_size n, double* L, blas_size ldl, double* b, blas_size incb, double c, double* Lo, blas_size ldlo) {
    blas_size one = 1;
    double one_d = 1.0;

	if(L != Lo || ldl != ldlo) {
		dlacpy("L", &n, &n, L, &ldl, Lo, &ldlo);
	}

    dcopy(&n, b, &incb, Lo + n, &ldlo);
	dtrsm("R", "L", "C", "N", &one, &n, &one_d, Lo, &ldlo, Lo + n, &ldlo);

    double border_inner = ddot(&n, Lo + n, &ldlo, Lo + n, &ldlo);
    Lo[n * ldlo + n] = sqrt(c - border_inner);
}

void cholesky_append_inplace_d(blas_size n, double* L, blas_size ldl) {
	assert(ldl >= n + 1);

	blas_size one = 1;
	double one_d = 1.0;

	dtrsm("R", "L", "C", "N", &one, &n, &one_d, L, &ldl, L + n, &ldl);
    double border_inner = ddot(&n, L + n, &ldl, L + n, &ldl);
	L[n * ldl + n] = sqrt(L[n * ldl + n] - border_inner);
}