#ifndef WENDA_CHOLESKY_UTILS_H_INCLUDED
#define WENDA_CHOLESKY_UTILS_H_INCLUDED

/*! Update the Cholesky representation of a given matrix.
 *
 *  This function computes the update of the cholesky decomposition R
 *  by a rank one perturbation given by x.
 */
void cholesky_update_d(int n, double* L, int ldl, double* x, int incx);

void cholesky_delete_d(int n, int i, double* L, int ldl, double* Lo, int lodl);

#endif