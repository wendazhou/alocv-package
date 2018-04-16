from alocv import _lasso as lasso, _helper_c as native_impl

import pytest
import numpy as np
import sklearn.linear_model as linear_model


def make_test_case(n, p, k, seed=42):
    random = np.random.RandomState(seed)

    X = random.randn(p, n).T
    beta = random.randn(p) * np.concatenate((np.ones(k), np.zeros(p - k)))
    random.shuffle(beta)
    y = np.dot(X, beta) + random.randn(n) * 0.1

    return X, y


def test_compute_h():
    X, y = make_test_case(100, 20, 10)

    E = np.concatenate((np.zeros(10), np.ones(10))).astype(np.bool)

    h_computed = lasso.compute_h_lasso(X, E)
    h_naive = lasso.compute_h_lasso_naive(X, E)

    assert np.linalg.norm(h_computed - h_naive) < 0.05


def test_compute_alo_path():
    X, y = make_test_case(50, 20, 10)

    alphas, beta_hats, _ = linear_model.lasso_path(X, y)
    alo = lasso.compute_alo_lasso_reference(X, y, beta_hats)

    beta_hat = beta_hats[:, 5]
    h_5 = lasso.compute_h_lasso(X, np.abs(beta_hat) > 1e-5)
    r_5 = y - np.dot(X, beta_hat)
    alo_5 = np.square(r_5 / (1 - h_5)).mean()

    assert len(alo) == len(alphas)
    assert np.allclose(alo_5, alo[5])


@pytest.mark.parametrize("method", [lasso._update_cholesky, native_impl.lasso_update_cholesky])
def test_update_cholesky_single(method):
    X, y = make_test_case(50, 5, 2)

    E = np.array([True, False, True, False, False])
    E_new = np.array([True, False, True, True, False])

    L = lasso._compute_cholesky(X, E)
    L_new, index_new = method(X, L, np.flatnonzero(E), np.flatnonzero(E_new))
    L_new_truth = lasso._compute_cholesky(X, E_new)

    assert np.linalg.norm(
        lasso._compute_leverage_cholesky(X, L_new, index_new)
        - lasso._compute_leverage_cholesky(X, L_new_truth, E_new)) < 0.001


@pytest.mark.parametrize("method", [lasso._update_cholesky, native_impl.lasso_update_cholesky])
def test_update_cholesky_no_order(method):
    X, y = make_test_case(50, 5, 2)

    E = np.array([True, False, True, False, False])
    E_new = np.array([True, True, True, False, False])

    perm = np.array([0, 2, 1, 3, 4])

    L = lasso._compute_cholesky(X, E)
    L_new, index_new = method(X, L, np.flatnonzero(E), np.flatnonzero(E_new))
    L_new_truth = lasso._compute_cholesky(X, E_new)
    L_new_truth_perm = lasso._compute_cholesky(X[:, perm], E_new[perm])

    assert np.linalg.norm(L_new - L_new_truth_perm) < 1e-5
    assert np.linalg.norm(
        lasso._compute_leverage_cholesky(X, L_new, index_new)
        - lasso._compute_leverage_cholesky(X, L_new_truth, E_new)) < 1e-3


@pytest.mark.parametrize("method", [lasso._update_cholesky, native_impl.lasso_update_cholesky])
def test_update_cholesky_multiple(method):
    X, y = make_test_case(50, 5, 2)

    E = np.array([True, False, True, False, False])
    E_new = np.array([True, True, True, True, False])

    L = lasso._compute_cholesky(X, E)
    L_new, index_new = method(X, L, np.flatnonzero(E), np.flatnonzero(E_new))
    L_new_truth = lasso._compute_cholesky(X, E_new)

    assert np.linalg.norm(
        lasso._compute_leverage_cholesky(X, L_new, index_new)
        - lasso._compute_leverage_cholesky(X, L_new_truth, E_new)) < 0.01


@pytest.mark.parametrize("method", [lasso._update_cholesky, native_impl.lasso_update_cholesky])
def test_update_cholesky_remove(method):
    X, y = make_test_case(50, 5, 2)

    E = np.array([True, True, True, False, False])
    E_new = np.array([True, False, True, False, False])

    L = lasso._compute_cholesky(X, E)
    L_new, index_new = method(X, L, np.flatnonzero(E), np.flatnonzero(E_new))
    L_new_truth = lasso._compute_cholesky(X, E_new)

    assert np.linalg.norm(
        lasso._compute_leverage_cholesky(X, L_new, index_new)
        - lasso._compute_leverage_cholesky(X, L_new_truth, E_new)) < 1e-5


@pytest.mark.parametrize("method", [lasso._update_cholesky, native_impl.lasso_update_cholesky])
def test_update_cholesky_mixed(method):
    X, y = make_test_case(50, 5, 2)

    E = np.array([True, False, True, True, False])
    E_new = np.array([True, True, True, False, False])

    L = lasso._compute_cholesky(X, E)
    L_new, index_new = method(X, L, np.flatnonzero(E), np.flatnonzero(E_new))
    L_new_truth = lasso._compute_cholesky(X, E_new)

    assert np.linalg.norm(
        lasso._compute_leverage_cholesky(X, L_new, index_new)
        - lasso._compute_leverage_cholesky(X, L_new_truth, E_new)) < 1e-5


def test_compute_alo_path_fast():
    X, y = make_test_case(50, 20, 10)

    alphas, beta_hats, _ = linear_model.lasso_path(X, y)

    alo = lasso.compute_alo_lasso_reference(X, y, beta_hats)
    alo_fast = lasso.compute_alo_lasso(X, y, beta_hats)

    assert np.all(np.isfinite(alo) == np.isfinite(alo_fast))
    assert np.square(alo[np.isfinite(alo)] - alo_fast[np.isfinite(alo_fast)]).mean() < 1e-3