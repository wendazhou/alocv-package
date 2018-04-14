from scipy.linalg.cython_blas cimport drotg, drot
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _cholupdate_f64(double[::1, :] L, double[::1] x) nogil:
    cdef int k
    cdef double c, s
    cdef int n, lenx
    cdef int inc = 1

    lenx = len(x)

    for k in range(lenx):
        drotg(&L[k, k], &x[k], &c, &s)
        n = lenx - k - 1
        drot(&n, &L[k + 1, k], &inc, &x[k + 1], &inc, &c, &s)


cpdef cholupdate_cythonblas(L, x):
    cdef np.ndarray[double, ndim=2] out = np.copy(L, order='F')
    cdef np.ndarray[double, ndim=1] x_copy = np.copy(x)

    _cholupdate_f64(out, x_copy)
    return out
