from alocv import _lasso as lasso

import numpy as np


def make_test_case(n, p, k):
    X = np.random.randn(n, p)
    beta = np.random.randn(p) * np.concatenate((np.ones(k), np.zeros(p - k)))
    np.random.shuffle(beta)
    y = np.dot(X, beta) + np.random.randn(n) * 0.1

    return X, y


def test_compute_h():
    X, y = make_test_case(100, 20, 10)

    E = np.concatenate((np.zeros(10), np.ones(10))).astype(np.bool)

    h_computed = lasso.compute_h_lasso(X, E)
    h_naive = lasso.compute_h_lasso_naive(X, E)

    assert np.linalg.norm(h_computed - h_naive) < 0.05
