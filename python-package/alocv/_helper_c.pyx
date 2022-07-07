#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from cython cimport view

########################################
# Cholesky utilities
########################################

cdef extern from "alocv/cholesky_utils.h":
    cdef void cholesky_update_d(int n, double* L, int ldl, double * x, int incx) nogil
    cdef void cholesky_downdate_d(int n, double* L, int ldl, double* x, int incx) nogil
    cdef void cholesky_delete_d(int n, int i, double* L, int ldl, double* Lo, int lodl) nogil
    cdef void cholesky_append_d(int n, double* L, int ldl, double* b, int incb, double c, double* Lo, int ldlo) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _cholupdate_d(double[::view.contiguous, :] L, double[:] x) nogil:
    cdef int ldl = L.strides[1] // sizeof(double)
    cdef int incx = x.strides[0] // sizeof(double)

    cholesky_update_d(len(x), &L[0, 0], ldl, &x[0], incx)


@cython.embedsignature(True)
cpdef cholupdate(L, x):
    """  Computes the Cholesky update of a given decomposition by a rank 1 update.
    
    Parameters
    ----------
    L: The lower-triangular Cholesky decomposition to update
    x: The rank-1 perturbation.
    """
    cdef np.ndarray[double, ndim=2] out = np.copy(L, order='F')
    cdef np.ndarray[double, ndim=1] x_copy = np.copy(x)

    _cholupdate_d(out, x_copy)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _choldowndate_d(double[::view.contiguous, :] L, double[:] x) nogil:
    cdef int ldl = L.strides[1] // sizeof(double)
    cdef int incx = x.strides[0] // sizeof(double)
    cholesky_downdate_d(len(x), &L[0, 0], ldl, &x[0], incx)


@cython.embedsignature(True)
def choldowndate(L, x, overwrite_x=False, overwrite_L=False):
    if not overwrite_x:
        x = np.copy(x)

    if not overwrite_L:
        L = np.copy(L, order='F')

    _choldowndate_d(L, x)
    return L


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _choldelete_d(double[::view.contiguous, :] L, double[::view.contiguous, :] Lo, int i) nogil:
    cdef int n = L.shape[0]
    cdef int ldl = L.strides[1] // sizeof(double)
    cdef int lodl = Lo.strides[1] // sizeof(double)

    cholesky_delete_d(n, i, &L[0, 0], ldl, &Lo[0, 0], lodl)


@cython.embedsignature(True)
cpdef choldelete(L, i, out=None):
    """ Updates the Cholesky decomposition when deleting a single column.
    
    Parameters
    ----------
    L: The lower-triangular Cholesky decomposition to update.
    i: The index of the location to delete.
    """
    n = L.shape[0]

    if out is None:
        out = np.zeros((n - 1, n - 1), order='F')

    _choldelete_d(L, out, i)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _cholappend_d(double[::view.contiguous, :] L, double[::view.contiguous, :] Lo, double[:] b, double c) nogil:
    cdef int n = L.shape[0]
    cdef int ldl = L.strides[1] // sizeof(double)
    cdef int ldlo = Lo.strides[1] // sizeof(double)
    cdef int incb = b.strides[0] // sizeof(double)

    cholesky_append_d(n, &L[0, 0], ldl, &b[0], incb, c, &Lo[0, 0], ldlo)

@cython.embedsignature(True)
def cholappend(L, b, c, out=None):
    """ Update the Cholesky decomposition when appending a single column.

    Parameters
    ----------
    L: the existing lower-triangular Cholesky decomposition to update
    b: a vector corresponding to border of the appended column.
    c: a scalar corresponding to the value of the appended corner.
    """
    n = L.shape[0]

    if out is None:
        out = np.zeros((n + 1, n + 1), order='F')

    _cholappend_d(L, out, b, c)
    return out


###########################################
# Lasso implementation
###########################################



cdef extern from "alocv/alo_lasso.h":
    cdef void lasso_compute_leverage_cholesky_d(int n, int k, double* W, int ldw, double* L, int ldl, double* leverage) nogil
    cdef void lasso_update_cholesky_w_d(int n, double* A, int lda, double* L, int ldl,
                                        double* W, int ldw, int len_index, int* index,
                                        int len_index_new, int* index_new) nogil
    cdef void lasso_compute_alo_d(int n, int p, int num_tuning, double* A, int lda, double* B, int ldb,
                                  double* y, int incy, const double* intercept, double tolerance, double* alo, double* leverage) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _lasso_update_cholesky_w_d(double[::view.contiguous, :] A,
                                     double[::view.contiguous, :] L,
                                     double[::view.contiguous, :] W,
                                     int[::1] index, int[::1] index_new) nogil:
    cdef int n = A.shape[0]
    cdef int lda = A.strides[1] // sizeof(double)
    cdef int ldl = L.strides[1] // sizeof(double)
    cdef int ldw = W.strides[1] // sizeof(double)

    lasso_update_cholesky_w_d(n, &A[0, 0], lda, &L[0, 0], ldl, &W[0, 0], ldw,
                              len(index), &index[0], len(index_new), &index_new[0])

@cython.embedsignature(True)
def lasso_update_cholesky_w(X, L, index, index_new, W=None, overwrite_L=False):
    cdef int[::1] index_view = np.array(index, dtype=np.int32)
    cdef int[::1] index_new_view = np.array(index_new, dtype=np.int32)

    n = X.shape[0]
    k_old = len(index)
    k_new = len(index_new_view)
    k_max = max(k_old, k_new)

    if not overwrite_L:
        L_old = L
        L = np.empty((k_max, k_max), order='F')
        np.copyto(L[:k_old, :k_old], L_old)


    if W is None:
        W = np.empty((n, k_new), order='F')

    _lasso_update_cholesky_w_d(X, L, W, index_view, index_new_view)

    if k_new > k_old and overwrite_L:
        # access base to subset correctly
        L = L.base

    return L[:k_new, :k_new], index_new_view


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _lasso_compute_leverage_cholesky(double[::view.contiguous, :] W,
                                           double[::view.contiguous, :] L,
                                           double[::1] leverage) nogil:
    cdef int n = W.shape[0]
    cdef int k = W.shape[1]
    cdef int ldw = W.strides[1] // sizeof(double)
    cdef int ldl = L.strides[1] // sizeof(double)

    lasso_compute_leverage_cholesky_d(n, k, &W[0, 0], ldw, &L[0, 0], ldl, &leverage[0])

@cython.embedsignature(True)
def lasso_compute_leverage_cholesky(A, L, index=None, out=None):
    if out is None:
        out = np.empty(A.shape[0])

    if index is not None:
        A = A[:, index]

    _lasso_compute_leverage_cholesky(A, L, out)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _lasso_compute_alo_d(double[::view.contiguous, :] A,
                               double[:] y,
                               double[::view.contiguous, :] B,
                               double tolerance,
                               double[::1] alo, double[::1, :] leverage) nogil:
    cdef int n = A.shape[0]
    cdef int p = A.shape[1]
    cdef int num_tuning = B.shape[1]
    cdef int lda = A.strides[1] // sizeof(double)
    cdef int ldb = B.strides[1] // sizeof(double)
    cdef int incy = y.strides[0] // sizeof(double)

    lasso_compute_alo_d(n, p, num_tuning, &A[0, 0], lda, &B[0, 0], ldb,
                        &y[0], incy, NULL, tolerance, &alo[0],
                        &leverage[0, 0] if leverage.shape[0] > 0 else NULL)


@cython.embedsignature(True)
def lasso_compute_alo(np.ndarray[double, ndim=2] X, double[:] y,
                      np.ndarray[double, ndim=2] beta_hats, double tolerance=1e-5,
                      out=None, return_leverage=False):
    """ Compute the ALO estimate for the LASSO.

    Parameters
    ----------
    X: The sensing matrix.
    y: The observation vector.
    beta_hats: The fitted coefficients.
    tolerance: A scalar value indicating the tolerance.
    out: Optional. If not None, a vector to store the ALO values.
    """
    if out is None:
        out = np.empty(beta_hats.shape[1])

    if not X.flags.f_contiguous:
        X = np.copy(X, order='F')

    if not beta_hats.flags.f_contiguous:
        beta_hats = np.copy(beta_hats, order='F')

    if return_leverage:
        leverage = np.empty((X.shape[0], beta_hats.shape[1]), order='F')
    else:
        leverage = np.empty((0, 0), order='F')

    _lasso_compute_alo_d(X, y, beta_hats, tolerance, out, leverage)

    if return_leverage:
        return out, leverage
    else:
        return out
