#ifndef WENDA_ALO_LASSO_H_INCLUDED
#define WENDA_ALO_LASSO_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

void lasso_update_cholesky_d(int n, double* A, int lda, double* L, int ldl, double* Lo, int lodl, int len_index, int* index, int len_index_new, int* index_new);

#ifdef __cplusplus
}
#endif

#endif // WENDA_ALO_LASSO_H_INCLUDED