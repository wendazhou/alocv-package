from alocv import _cholesky


import numpy as np
import scipy.linalg


def test_givens():
    a = 1
    b = 2

    c, s, r = _cholesky._givens(a, b)

    assert r == np.hypot(a, b)
    assert c * a + s * b == r


def test_cholupdate_upper():
    random = np.random.RandomState(42)
    p = 500
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)
    x_update = random.randn(p)

    L = scipy.linalg.cholesky(S, lower=False)

    S_update = S + np.outer(x_update, x_update)

    L_update_truth = scipy.linalg.cholesky(S_update, lower=False)
    L_update = _cholesky.cholupdate(L, x_update, upper=True)

    assert np.linalg.norm(np.dot(L_update_truth.T, L_update_truth) - S_update) < 0.01
    assert np.linalg.norm(np.dot(L_update.T, L_update) - S_update) < 0.01
    assert np.linalg.norm(np.triu(L_update) - np.triu(L_update_truth), 'fro') < 0.05


def test_cholupdate_lower():
    random = np.random.RandomState(42)
    p = 3
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)
    x_update = random.randn(p)

    L = scipy.linalg.cholesky(S, lower=True)

    S_update = S + np.outer(x_update, x_update)

    L_update_truth = scipy.linalg.cholesky(S_update, lower=True).T
    L_update = _cholesky.cholupdate(L, x_update, upper=False).T

    assert np.linalg.norm(np.dot(L_update_truth.T, L_update_truth) - S_update) < 0.01
    assert np.linalg.norm(np.dot(L_update.T, L_update) - S_update) < 0.01
    assert np.linalg.norm(np.triu(L_update) - np.triu(L_update_truth), 'fro') < 0.05


def test_cholupdate_blas():
    random = np.random.RandomState(42)
    p = 500
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)
    x_update = random.randn(p)

    L = scipy.linalg.cholesky(S, lower=True)

    S_update = S + np.outer(x_update, x_update)

    L_update_truth = scipy.linalg.cholesky(S_update, lower=True).T
    L_update = _cholesky.cholupdate_blas(L, x_update).T

    assert np.linalg.norm(np.dot(L_update_truth.T, L_update_truth) - S_update) < 0.01
    assert np.linalg.norm(np.dot(L_update.T, L_update) - S_update) < 0.01
    assert np.linalg.norm(np.triu(L_update) - np.triu(L_update_truth), 'fro') < 0.05


def test_cholupdate_blas_upper():
    random = np.random.RandomState(42)
    p = 500
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)
    x_update = random.randn(p)

    L = scipy.linalg.cholesky(S, lower=False)

    S_update = S + np.outer(x_update, x_update)

    L_update_truth = scipy.linalg.cholesky(S_update, lower=False)
    L_update = _cholesky.cholupdate_blas(L, x_update, upper=True)

    assert np.linalg.norm(np.dot(L_update_truth.T, L_update_truth) - S_update) < 0.01
    assert np.linalg.norm(np.dot(L_update.T, L_update) - S_update) < 0.01
    assert np.linalg.norm(np.triu(L_update) - np.triu(L_update_truth), 'fro') < 0.05
