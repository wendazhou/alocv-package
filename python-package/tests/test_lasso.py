from alocv import _lasso as lasso

import numpy as np
import sklearn.linear_model as linear_model


def make_test_case(n, p, k):
    X = np.random.randn(p, n).T
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


def test_compute_alo_path():
    X, y = make_test_case(100, 20, 10)

    alphas, beta_hats, _ = linear_model.lasso_path(X, y)
    alo = lasso.compute_alo_lasso_reference(X, y, beta_hats)

    assert len(alo) == len(alphas)


def test_update_cholesky_single():
    X, y = make_test_case(50, 5, 2)

    E = np.array([True, False, False, False, False])
    E_new = np.array([True, False, True, False, False])

    L = lasso.compute_cholesky(X, E)
    L_new, index_new = lasso.update_cholesky(X, L, np.flatnonzero(E), np.flatnonzero(E_new))
    L_new_truth = lasso.compute_cholesky(X, E_new)

    assert np.linalg.norm(
        lasso.compute_leverage_cholesky(X, L_new, index_new)
        - lasso.compute_leverage_cholesky(X, L_new_truth, E_new)) < 0.01


def test_update_cholesky_no_order():
    X, y = make_test_case(50, 5, 2)

    E = np.array([True, False, True, False, False])
    E_new = np.array([True, True, True, False, False])

    L = lasso.compute_cholesky(X, E)
    L_new, index_new = lasso.update_cholesky(X, L, np.flatnonzero(E), np.flatnonzero(E_new))
    L_new_truth = lasso.compute_cholesky(X, E_new)

    assert np.linalg.norm(
        lasso.compute_leverage_cholesky(X, L_new, index_new)
        - lasso.compute_leverage_cholesky(X, L_new_truth, E_new)) < 0.01


def test_update_cholesky_multiple():
    X, y = make_test_case(50, 5, 2)

    E = np.array([True, False, True, False, False])
    E_new = np.array([True, True, True, True, False])

    L = lasso.compute_cholesky(X, E)
    L_new, index_new = lasso.update_cholesky(X, L, np.flatnonzero(E), np.flatnonzero(E_new))
    L_new_truth = lasso.compute_cholesky(X, E_new)

    assert np.linalg.norm(
        lasso.compute_leverage_cholesky(X, L_new, index_new)
        - lasso.compute_leverage_cholesky(X, L_new_truth, E_new)) < 0.01
