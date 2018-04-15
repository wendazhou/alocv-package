import numpy as np
cimport numpy as np
cimport cython
from cython cimport view

cdef extern from "alocv/cholesky_utils.h":
    cdef void cholesky_update_d(int n, double* L, int ldl, double * x, int incx) nogil
    cdef void cholesky_delete_d(int n, int i, double* L, int ldl, double* Lo, int lodl) nogil
    cdef void cholesky_append_d(int n, double* L, int ldl, double* b, int incb, double c, double* Lo, int lodl) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
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
cdef void _choldelete_d(double[::view.contiguous, :] L, double[::view.contiguous, :] Lo, int i) nogil:
    cdef int n = L.shape[0]
    cdef int ldl = L.strides[1] // sizeof(double)
    cdef int lodl = Lo.strides[1] // sizeof(double)

    cholesky_delete_d(n, i, &L[0, 0], ldl, &Lo[0, 0], lodl)


@cython.embedsignature(True)
cpdef choldelete(L, i):
    """ Updates the Cholesky decomposition when deleting a single column.
    
    Parameters
    ----------
    L: The lower-triangular Cholesky decomposition to update.
    i: The index of the location to delete.
    """
    n = L.shape[0]
    cdef np.ndarray[double, ndim=2] out = np.zeros((n - 1, n - 1), order='F')
    _choldelete_d(L, out, i)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _cholappend_d(double[::view.contiguous, :] L, double[::view.contiguous, :] Lo, double[:] b, double c) nogil:
    cdef int n = L.shape[0]
    cdef int ldl = L.strides[1] // sizeof(double)
    cdef int lodl = Lo.strides[1] // sizeof(double)
    cdef int incb = b.strides[0] // sizeof(double)

    cholesky_append_d(n, &L[0, 0], ldl, &b[0], incb, c, &Lo[0, 0], lodl)

@cython.embedsignature(True)
def cholappend(L, b, c):
    """ Update the Cholesky decomposition when appending a single column.

    Parameters
    ----------
    L: the existing lower-triangular Cholesky decomposition to update
    b: a vector corresponding to border of the appended column.
    c: a scalar corresponding to the value of the appended corner.
    """
    n = L.shape[0]
    cdef np.ndarray[double, ndim=2] out = np.zeros((n + 1, n + 1), order='F')
    _cholappend_d(L, out, b, c)

    return out
