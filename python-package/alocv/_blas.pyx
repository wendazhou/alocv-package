#cython: language_level=3

cimport scipy.linalg.cython_blas
cimport scipy.linalg.cython_lapack


cdef extern from "alocv/alo_config.h":
    ctypedef int blas_size

cdef public void ALOCV_blas_daxpy(const blas_size *n, const double *alpha, const double *x, const blas_size *incx, double *y,
                                  const blas_size *incy) nogil:
    scipy.linalg.cython_blas.daxpy(n, alpha, x, incx, y, incy)
cdef public void ALOCV_blas_dcopy(const blas_size *n, const double *x, const blas_size *incx, double *y, const blas_size *incy) nogil:
    scipy.linalg.cython_blas.dcopy(n, x, incx, y, incy)
cdef public double ALOCV_blas_ddot(const blas_size *n, const double *x, const blas_size *incx, const double *y, const blas_size *incy) nogil:
    return scipy.linalg.cython_blas.ddot(n, x, incx, y, incy)
cdef public void ALOCV_blas_dgemm(const char *transa, const char *transb, const blas_size *m, const blas_size *n, const blas_size *k,
           const double *alpha, const double *a, const blas_size *lda, const double *b, const blas_size *ldb,
           const double *beta, double *c, const blas_size *ldc) nogil:
    scipy.linalg.cython_blas.dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
cdef public void ALOCV_blas_dgemv(const char *trans, const blas_size *m, const blas_size *n, const double *alpha, const double *a,
           const blas_size *lda, const double *x, const blas_size *incx, const double *beta, double *y,
           const blas_size *incy) nogil:
    scipy.linalg.cython_blas.dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
cdef public void ALOCV_blas_dgeqrf(const blas_size *m, const blas_size *n, double *a, const blas_size *lda, double *tau, double *work,
            const blas_size *lwork, blas_size *info) nogil:
    scipy.linalg.cython_lapack.dgeqrf(m, n, a, lda, tau, work, lwork, info)
cdef public void ALOCV_blas_dlacpy(const char *uplo, const blas_size *m, const blas_size *n, const double *a, const blas_size *lda, double *b,
            const blas_size *ldb) nogil:
    scipy.linalg.cython_lapack.dlacpy(uplo, m, n, a, lda, b, ldb)
cdef public void ALOCV_blas_dormqr(const char *side, const char *trans, const blas_size *m, const blas_size *n, const blas_size *k,
            const double *a, const blas_size *lda, const double *tau, double *c, const blas_size *ldc, double *work,
            const blas_size *lwork, blas_size *info) nogil:
    scipy.linalg.cython_lapack.dormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info)
cdef public void ALOCV_blas_dpotrf(const char *uplo, const blas_size *n, double *a, const blas_size *lda, blas_size *info) nogil:
    scipy.linalg.cython_lapack.dpotrf(uplo, n, a, lda, info)
cdef public void ALOCV_blas_dpstrf(const char *uplo, const blas_size *n, double *a, const blas_size *lda, blas_size *piv, blas_size *rank,
            const double *tol, double *work, blas_size *info) nogil:
    scipy.linalg.cython_lapack.dpstrf(uplo, n, a, lda, piv, rank, tol, work, info)
cdef public void ALOCV_blas_drot(const blas_size *n, double *x, const blas_size *incx, double *y, const blas_size *incy, const double *c,
          const double *s) nogil:
    scipy.linalg.cython_blas.drot(n, x, incx, y, incy, c, s)
cdef public void ALOCV_blas_drotg(double *a, double *b, double *c, double *s) nogil:
    scipy.linalg.cython_blas.drotg(a, b, c, s)
cdef public void ALOCV_blas_dscal(const blas_size *n, const double *a, double *x, const blas_size *incx) nogil:
    scipy.linalg.cython_blas.dscal(n, a, x, incx)
cdef public void ALOCV_blas_dsymm(const char *side, const char *uplo, const blas_size *m, const blas_size *n, const double *alpha,
           const double *a, const blas_size *lda, const double *b, const blas_size *ldb, const double *beta, double *c,
           const blas_size *ldc) nogil:
    scipy.linalg.cython_blas.dsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
cdef public void ALOCV_blas_dsyrk(const char *uplo, const char *trans, const blas_size *n, const blas_size *k, const double *alpha,
           const double *a, const blas_size *lda, const double *beta, double *c, const blas_size *ldc) nogil:
    scipy.linalg.cython_blas.dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
cdef public void ALOCV_blas_dtrmm(const char *side, const char *uplo, const char *transa, const char *diag, const blas_size *m,
           const blas_size *n, const double *alpha, const double *a, const blas_size *lda, double *b,
           const blas_size *ldb) nogil:
    scipy.linalg.cython_blas.dtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
cdef public void ALOCV_blas_dtrsm(const char *side, const char *uplo, const char *transa, const char *diag, const blas_size *m,
           const blas_size *n, const double *alpha, const double *a, const blas_size *lda, double *b,
           const blas_size *ldb) nogil:
    scipy.linalg.cython_blas.dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
cdef public void ALOCV_blas_dtrtri(const char *uplo, const char *diag, const blas_size *n, double *a, const blas_size *lda, blas_size *info) nogil:
    scipy.linalg.cython_lapack.dtrtri(uplo, diag, n, a, lda, info)
    
