""" Functions to compute the ALO for lasso.

"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular, solve

from ._helper_c import cholappend, choldelete


def compute_alo_lasso_reference(X, y, beta_hats):
    """ Compute ALO approximation of error for lasso.

    Parameters
    ----------
    X: the sensing matrix
    y: the observations
    beta_hats: A matrix [num_observations, num_alphas] of fits for a number
        of tuning along the path.

    Returns
    -------
    alo_values: The ALO mean-squared error for each tuning.
    """
    residuals = y[:, np.newaxis] - np.dot(X, beta_hats)

    alo_values = np.empty(beta_hats.shape[1])

    for i in range(beta_hats.shape[1]):
        E = np.abs(beta_hats[:, i]) > 1e-5
        h = compute_h_lasso(X, E)
        alo_values[i] = np.square(np.divide(residuals[:, i], (1 - h))).mean()

    return alo_values


def _compute_cholesky(X, index):
    S = np.dot(X[:, index].T, X[:, index])
    return cholesky(S, lower=True, overwrite_a=True, check_finite=False)


def _update_cholesky(X, L, index, index_new):
    index_added = set(index_new) - set(index)
    index_removed = set(index) - set(index_new)

    index = list(index)

    for i in index_removed:
        loc = index.index(i)
        del index[loc]
        L = choldelete(L, i)

    for i in index_added:
        L = cholappend(L, np.dot(X[:, index].T, X[:, i]), np.dot(X[:, i].T, X[:, i]))
        index.append(i)

    return L, index


def _compute_leverage_cholesky(X, L, index):
    W = X[:, index]
    return np.sum(solve_triangular(L, W.T, lower=True, check_finite=False) ** 2, axis=0)


def compute_alo_lasso(X, y, beta_hats):
    residuals = y[:, np.newaxis] - np.dot(X, beta_hats)
    alo_values = np.empty(beta_hats.shape[1])

    cholesky_current = None
    active_index = None

    for i in range(beta_hats.shape[1]):
        E = np.abs(beta_hats[:, i]) > 1e-5
        num_active = np.sum(E)

        if num_active == 0:
            cholesky_current = None
            active_index = []
            alo_values[i] = np.mean(np.square(residuals[:, i]))
            continue

        if cholesky_current is None:
            current_index = np.flatnonzero(E)
            cholesky_current = _compute_cholesky(X, current_index)
        else:
            current_index = np.flatnonzero(E)
            cholesky_current, current_index = _update_cholesky(X, cholesky_current, active_index, current_index)

        active_index = current_index

        if num_active >= X.shape[0]:
            leverage = np.ones(X.shape[0], dtype=X.dtype)
        else:
            leverage = _compute_leverage_cholesky(X, cholesky_current, active_index)

        alo_values[i] = np.mean(np.square(residuals[:, i] / (1 - leverage)))

    return alo_values


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
    num_active = np.sum(E)

    if num_active == 0:
        return np.zeros(X.shape[0], dtype=X.dtype)
    elif num_active >= X.shape[0]:
        return np.ones(X.shape[0], dtype=X.dtype)

    W = X[:, E]
    S = np.dot(W.T, W)
    K = cholesky(S, lower=True, overwrite_a=True, check_finite=False)
    return np.sum(solve_triangular(K, W.T, lower=True, check_finite=False) ** 2, axis=0)


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
