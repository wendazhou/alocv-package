#ifndef WENDA_CHOLESKY_UTILS_H_INCLUDED
#define WENDA_CHOLESKY_UTILS_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

/*! Update the Cholesky representation of a given matrix.
 *
 *  This function computes the update of the cholesky decomposition L
 *  by a rank one perturbation given by x.
 */
void cholesky_update_d(int n, double* L, int ldl, double* x, int incx);

/*! Update the Cholesky representation when deleting a column.
 *
 * This function computes the update of the Cholesky decomposition L
 * when deleting a single column at location i from the original matrix.
 */
void cholesky_delete_d(int n, int i, double* L, int ldl, double* Lo, int lodl);

/*! Update the Cholesky representation when adding o column.
 *
 * This function computes the update of the Cholesky decomposition L
 * when adding a single column at the end of the original matrix.
 */
void cholesky_append_d(int n, double* L, int ldl, double* b, int incb, double c, double* Lo, int lodl);

#ifdef __cplusplus
}
#endif

#endif