#pragma once

#include <alocv/alo_config.h>

#ifndef ALOCV_LAPACK_NO_RFP
#define ALOCV_LAPACK_NO_RFP
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ALOCV_CALL(name) ALOCV_blas_##name

#define drot ALOCV_CALL(drot)
#define drotg ALOCV_CALL(drotg)
#define daxpy ALOCV_CALL(daxpy)
#define dscal ALOCV_CALL(dscal)
#define dlacpy ALOCV_CALL(dlacpy)
#define dcopy ALOCV_CALL(dcopy)
#define dgemm ALOCV_CALL(dgemm)
#define dtrsm ALOCV_CALL(dtrsm)
#define dtrmm ALOCV_CALL(dtrmm)
#define dsymm ALOCV_CALL(dsymm)
#define ddot ALOCV_CALL(ddot)
#define dgemv ALOCV_CALL(dgemv)
#define dsymv ALOCV_CALL(dsymv)
#define dpotrf ALOCV_CALL(dpotrf)
#define dpstrf ALOCV_CALL(dpstrf)
#define dgels ALOCV_CALL(dgels)
#define dgeqrf ALOCV_CALL(dgeqrf)
#define dormqr ALOCV_CALL(dormqr)
#define dtrtri ALOCV_CALL(dtrtri)
#define dsyrk ALOCV_CALL(dsyrk)

void daxpy(const blas_size *n, const double *alpha, const double *x, const blas_size *incx, double *y,
           const blas_size *incy);
void dcopy(const blas_size *n, const double *x, const blas_size *incx, double *y, const blas_size *incy);
double ddot(const blas_size *n, const double *x, const blas_size *incx, const double *y, const blas_size *incy);
void dgemm(const char *transa, const char *transb, const blas_size *m, const blas_size *n, const blas_size *k,
           const double *alpha, const double *a, const blas_size *lda, const double *b, const blas_size *ldb,
           const double *beta, double *c, const blas_size *ldc);
void dgemv(const char *trans, const blas_size *m, const blas_size *n, const double *alpha, const double *a,
           const blas_size *lda, const double *x, const blas_size *incx, const double *beta, double *y,
           const blas_size *incy);
void dgeqrf(const blas_size *m, const blas_size *n, double *a, const blas_size *lda, double *tau, double *work,
            const blas_size *lwork, blas_size *info);
void dlacpy(const char *uplo, const blas_size *m, const blas_size *n, const double *a, const blas_size *lda, double *b,
            const blas_size *ldb);
void dormqr(const char *side, const char *trans, const blas_size *m, const blas_size *n, const blas_size *k,
            const double *a, const blas_size *lda, const double *tau, double *c, const blas_size *ldc, double *work,
            const blas_size *lwork, blas_size *info);
void dpotrf(const char *uplo, const blas_size *n, double *a, const blas_size *lda, blas_size *info);
void dpstrf(const char *uplo, const blas_size *n, double *a, const blas_size *lda, blas_size *piv, blas_size *rank,
            const double *tol, double *work, blas_size *info);

void drot(const blas_size *n, double *x, const blas_size *incx, double *y, const blas_size *incy, const double *c,
          const double *s);
void drotg(double *a, double *b, double *c, double *s);
void dscal(const blas_size *n, const double *a, double *x, const blas_size *incx);
void dsymm(const char *side, const char *uplo, const blas_size *m, const blas_size *n, const double *alpha,
           const double *a, const blas_size *lda, const double *b, const blas_size *ldb, const double *beta, double *c,
           const blas_size *ldc);
void dsyrk(const char *uplo, const char *trans, const blas_size *n, const blas_size *k, const double *alpha,
           const double *a, const blas_size *lda, const double *beta, double *c, const blas_size *ldc);
void dsymv(const char *uplo, const blas_size *n, const double *alpha, const double *a, const blas_size *lda,
           const double *x, const blas_size *incx, const double *beta, double *y, const blas_size *incy);

void dtrmm(const char *side, const char *uplo, const char *transa, const char *diag, const blas_size *m,
           const blas_size *n, const double *alpha, const double *a, const blas_size *lda, double *b,
           const blas_size *ldb);
void dtrsm(const char *side, const char *uplo, const char *transa, const char *diag, const blas_size *m,
           const blas_size *n, const double *alpha, const double *a, const blas_size *lda, double *b,
           const blas_size *ldb);
void dtrtri(const char *uplo, const char *diag, const blas_size *n, double *a, const blas_size *lda, blas_size *info);



#ifdef __cplusplus
}
#endif
