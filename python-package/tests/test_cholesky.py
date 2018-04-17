from alocv import _cholesky, _helper_c as native_impl

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


def test_cholupdate_cython():
    random = np.random.RandomState(42)
    p = 20
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)
    x_update = random.randn(p)

    L = scipy.linalg.cholesky(S, lower=True)

    S_update = S + np.outer(x_update, x_update)

    L_update_truth = scipy.linalg.cholesky(S_update, lower=True).T
    L_update = native_impl.cholupdate(L, x_update).T

    assert np.linalg.norm(np.dot(L_update_truth.T, L_update_truth) - S_update) < 1e-5
    assert np.linalg.norm(np.dot(L_update.T, L_update) - S_update) < 1e-5
    assert np.linalg.norm(np.triu(L_update) - np.triu(L_update_truth), 'fro') < 1e-5


def test_cholappend_upper():
    random = np.random.RandomState(42)
    p = 200
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)

    S_small = S[:-1, :-1]

    L = scipy.linalg.cholesky(S_small)

    L_large_truth = scipy.linalg.cholesky(S)
    L_large = _cholesky.cholappend(L, S[-1, :-1], S[-1, -1], upper=True)

    assert np.linalg.norm(np.triu(L_large) - np.triu(L_large_truth)) < 0.05


def test_cholappend_lower():
    random = np.random.RandomState(42)
    p = 200
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)

    S_small = S[:-1, :-1]

    L = scipy.linalg.cholesky(S_small, lower=True)

    L_large_truth = scipy.linalg.cholesky(S, lower=True)
    L_large = _cholesky.cholappend(L, S[-1, :-1], S[-1, -1], upper=False)

    assert np.linalg.norm(L_large - L_large_truth) < 0.05


def test_cholappend_cython():
    random = np.random.RandomState(42)
    p = 4
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)

    S_small = S[:-1, :-1]

    L = scipy.linalg.cholesky(S_small, lower=True)

    L_large_truth = scipy.linalg.cholesky(S, lower=True)
    L_large = native_impl.cholappend(L, S[-1, :-1], S[-1, -1])

    assert np.linalg.norm(np.triu(L_large) - np.triu(L_large_truth)) < 1e-3


def test_cholappend_cython_inplace():
    random = np.random.RandomState(42)
    p = 4
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)

    S_small = S[:-1, :-1]

    L = scipy.linalg.cholesky(S_small, lower=True)
    L_large_truth = scipy.linalg.cholesky(S, lower=True)

    L_large = np.zeros_like(L_large_truth)
    L_large[:-1, :-1] = L

    L_large_inplace = native_impl.cholappend(L_large[:-1, :-1], S[-1, :-1], S[-1, -1], out=L_large)

    assert L_large_inplace is L_large
    assert np.linalg.norm(np.triu(L_large) - np.triu(L_large_truth)) < 1e-3


def test_cholappend_cython_inplace_all():
    random = np.random.RandomState(42)
    p = 4
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)

    S_small = S[:-1, :-1]

    L = scipy.linalg.cholesky(S_small, lower=True)
    L_large_truth = scipy.linalg.cholesky(S, lower=True)

    L_large = np.zeros_like(L_large_truth)
    L_large[:-1, :-1] = L
    L_large[:, -1] = S[:, -1]

    L_large_inplace = native_impl.cholappend(L_large[:-1, :-1], L_large[:-1, -1], L_large[-1, -1], out=L_large)

    assert L_large_inplace is L_large
    assert np.linalg.norm(np.tril(L_large) - np.tril(L_large_truth)) < 1e-3


def test_choldelete_upper():
    random = np.random.RandomState(42)
    p = 5
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)

    E = np.ones(p, dtype=np.bool)
    E[p // 2] = False

    S_small = S[:, E][E, :]

    L = scipy.linalg.cholesky(S, lower=False)
    L_small_truth = scipy.linalg.cholesky(S_small, lower=False)
    L_small = _cholesky.choldelete(L, p // 2)

    assert np.linalg.norm(np.triu(L_small) - np.triu(L_small_truth)) < 0.05


def test_choldelete_cython():
    random = np.random.RandomState(42)
    p = 20
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)

    E = np.ones(p, dtype=np.bool)
    E[p // 2] = False

    S_small = S[:, E][E, :]

    L = scipy.linalg.cholesky(S, lower=True)
    L_small_truth = scipy.linalg.cholesky(S_small, lower=True)
    L_small = native_impl.choldelete(L, p // 2)

    assert np.linalg.norm(np.tril(L_small) - np.tril(L_small_truth)) < 0.001


def test_choldelete_cython_inplace():
    random = np.random.RandomState(42)
    p = 5
    X = random.randn(2 * p, p)
    S = np.dot(X.T, X)

    E = np.ones(p, dtype=np.bool)
    E[p // 2] = False

    L = scipy.linalg.cholesky(S, lower=True)
    L_small = native_impl.choldelete(L, p // 2)
    L_small_inplace = native_impl.choldelete(L, p // 2, out=L[:-1, :-1])

    assert L_small_inplace.base is L
    assert np.linalg.norm(np.tril(L_small) - np.tril(L_small_inplace)) < 0.001
