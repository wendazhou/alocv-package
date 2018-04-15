#include "alocv/cholesky_utils.h"
#include "blas_configuration.h"


void cholesky_update_d(int n, double* L, int ldl, double* x, int incx) {
    double c, s;
    int incr = 1;

    for(int k = 0; k < n; k++) {
        int m = n - k - 1;
        drotg(L + ldl * k + k, x + incx * k, &c, &s);
        drot(&m, L + ldl * k + k + 1, &incr,
            x + incx * (k + 1), &incx, &c, &s);
    }
}

void cholesky_delete_d(int n, int i, double* L, int ldl, double* Lo, int lodl) {
    int s22_length = n - i - 1;

    dlacpy("L", &i, &i, L, &ldl, Lo, &lodl);
    dlacpy("A", &s22_length, &i, L + i + 1, &ldl, Lo + i, &lodl);
    dlacpy("L", &s22_length, &s22_length, L + (i + 1) * ldl + (i + 1), &ldl, Lo + i * lodl + i, &lodl);
    cholesky_update_d(s22_length, Lo + i * lodl + i, lodl, L + i * ldl + (i + 1), 1);
}