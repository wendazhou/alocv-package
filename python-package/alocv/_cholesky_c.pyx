from scipy.linalg.cython_blas cimport drotg, drot
import numpy as np
cimport numpy as np
cimport cython
from cython cimport view


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _cholupdate_d(int n, double* L, int ldl, double* x, int incx) nogil:
    cdef double c, s
    cdef int k
    cdef int num_remain
    cdef int incr = 1

    for k in range(n):
        drotg(L + ldl * k + k, x + k * incx, &c, &s)
        num_remain = n - k - 1
        drot(&num_remain, L + ldl * k + k + 1, &incr, x + (k + 1) * incx, &incx, &c, &s)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _cholupdate_f64(double[::view.contiguous, :] L, double[:] x) nogil:
    cdef int ldl = L.strides[1] // sizeof(double)
    cdef int incx = x.strides[0] // sizeof(double)
    _cholupdate_d(len(x), &L[0, 0], ldl, &x[0], incx)


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

    _cholupdate_f64(out, x_copy)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _choldelete_d(double[::view.contiguous, :] L, double[::view.contiguous, :] Lo, int i) nogil:
    Lo[:i, :i] = L[:i, :i]
    Lo[i:, :i] = L[i + 1:, :i]
    Lo[i:, i:] = L[i + 1:, i + 1:]

    _cholupdate_f64(Lo[i:, i:], L[i + 1:, i])


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
