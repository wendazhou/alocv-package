""" Utilities for manipulating Cholesky functions.

"""

import numpy as np
import math
import scipy.linalg.blas as blas


def _givens(a, b):
    if math.fabs(a) > math.fabs(b):
        t = b / a
        u = math.copysign(math.sqrt(1 + t * t), a)
        c = 1 / u
        s = c * t
        r = a * u
    else:
        t = a / b
        u = math.copysign(math.sqrt(1 + t * t), b)
        s = 1 / u
        c = s * t
        r = b * u

    return c, s, r


def cholupdate(L, x, upper=False, overwrite_x=False, out=None):
    """ Updates the cholesky decomposition L by the rank one update
    given by x*x'

    Parameters
    ----------
    L: The original Cholesky decomposition to update.
    x: The vector to update the Cholesky decomposition by.
    upper: If True, indicate that the original decomposition was
        upper triangular.
    overwrite_x: Whether the function is allowed to overwrite the passed
        in x value
    out: If not None, the matrix in which to store the output.

    Returns
    -------
    The updated Cholesky decomposition.
    """
    if out is None:
        out = np.zeros_like(L)

    if not overwrite_x:
        x = np.copy(x)

    if not upper:
        L = L.T
        out = out.T

    for k in range(len(x)):
        l_k = L[k, k]
        x_k = x[k]

        c, s, r = _givens(l_k, x_k)

        out[k, k] = r
        out[k, k + 1:] = c * L[k, k + 1:] + s * x[k + 1:]
        x[k + 1:] = -s * L[k, k + 1:] + c * x[k + 1:]

    if not upper:
        out = out.T

    return out


def cholupdate_blas(L, x, upper=False, overwrite_x=False, out=None):
    if out is None:
        out = np.copy(L, order='C' if upper else 'F')
    elif out is not L:
        np.copyto(out, L)

    if not overwrite_x:
        x = np.copy(x)

    if upper:
        out = out.T

    rotg = blas.get_blas_funcs('rotg', (L,))
    rot = blas.get_blas_funcs('rot', (L, x))

    n = len(x)

    for k in range(n):
        c, s = rotg(out[k, k], x[k])
        rot(out, x, c, s, n=n - k, offx=k * n + k, offy=k, overwrite_x=True, overwrite_y=True)

    if upper:
        out = out.T

    return out
