""" Functions to compute the ALO for lasso.

"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular, solve


def compute_h_lasso(X, E):
    """ Compute leverage for Lasso estimator.

    Computes the leverage for the Lasso estimator with the given
    equicorellation set

    Parameters
    ----------
    X: The design matrix.
    E: The equicorellation set.

    Returns
    -------
    h: a vector representing the leverage values for each observation
    """
    W = X[:, E]
    S = np.dot(W.T, W)
    K = cholesky(S, overwrite_a=True, check_finite=False)
    return np.sum(solve_triangular(K, W.T, check_finite=False) ** 2, axis=0)


def compute_h_lasso_naive(X, E):
    """ Computes the leverage for the Lasso estimator using a naive method.

    Computes the leverage for the Lasso estimator with the given
    equicorellation set. See `compute_h_lasso` for a computationally
    efficient strategy to compute the leverage values.

    Parameters
    ----------
    X: The design matrix.
    E: The equicorellation set.

    Returns
    -------
    h: a vector representing the leverage values for each observation
    """
    W = X[:, E]
    S = np.dot(W.T, W)
    return np.diag(W.dot(solve(S, W.T, sym_pos=True)))
