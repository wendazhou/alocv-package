#include <alocv/alo_svm.h>
#include "blas_configuration.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include <vector>

#include "gram_utils.hpp"


namespace {

void svm_compute_gsub_impl(blas_size n, blas_size nv, const double* kv,
                           const double* K, blas_size ldk, const double* y,
                           const std::vector<blas_size>& s_idx, const std::vector<blas_size>& v_idx,
						   double* g_sub, double* g_temp) {
    double* kv2 = static_cast<double*>(blas_malloc(16, n * nv * sizeof(double)));

    blas_size one_i = 1;

    // create a second copy in kv2 (needed for solving linear system).
    std::copy(kv, kv + n * nv, kv2);

    double temp_size;
    blas_size info;
    blas_size min_one_i = -1;

    // before we do anything, we can compute the g vector
    // we start by computing the entries in V
    for(auto i : s_idx) {
        daxpy(&n, y + i, K + i * ldk, &one_i, g_temp, &one_i);
    }

    dgels("N", &n, &nv, &one_i, kv2, &n, g_temp, &n, &temp_size, &min_one_i, &info);

    blas_size work_size = static_cast<blas_size>(temp_size);
    double* work = static_cast<double*>(blas_malloc(16, work_size * sizeof(double)));

    dgels("N", &n, &nv, &one_i, kv2, &n, g_temp, &n, work, &work_size, &info);

    blas_free(work);
    blas_free(kv2);

    for(blas_size i = 0; i < v_idx.size(); ++i) {
        g_sub[v_idx[i]] = g_temp[i];
    }
}

}


void svm_compute_alo(blas_size n, const double* K, blas_size ldk, const double* y, const double* alpha,
                     double rho, double lambda, double tol, double* alo_predicted, double* alo_mse) {
    double* y_pred = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
    double* y_hat = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
    double* a_slack = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
    double* g_sub = static_cast<double*>(blas_malloc(16, n * sizeof(double)));

    blas_size one_i = 1;
    double one = 1.0;
    double min_one = -1.0;
    double zero = 0.0;

    // we compute K * alpha in y_pred
    dsymv("L", &n, &one, K, &ldk, alpha, &one_i, &zero, y_pred, &one_i);

    // a_slack will contain -lambda * y_hat
    std::transform(y_pred, y_pred + n, a_slack, [=](double x) { return -x * lambda; });

	// Add offset so that y_pred now contains y_hat
	std::transform(y_pred, y_pred + n, y_pred, [=](double x) { return x - rho; });

    // save the predicted values in y_hat
	std::copy(y_pred, y_pred + n, y_hat);

    // y_pred will contain y * y_hat
    std::transform(y_pred, y_pred + n, y, y_pred, std::multiplies<>{});

    std::vector<blas_size> v_idx;
    std::vector<blas_size> s_idx;

    for(blas_size i = 0; i < n; ++i) {
        if(std::abs(1 - y_pred[i]) < tol) {
            v_idx.push_back(i);
        } else if (y_pred[i] <= 1.0 - tol) {
            s_idx.push_back(i);
        }
    }

    blas_size nv = static_cast<blas_size>(v_idx.size());
    blas_size ns = static_cast<blas_size>(s_idx.size());

    double* kv = static_cast<double*>(blas_malloc(16, n * nv * sizeof(double)));

    // copy the respective columns of K into matrix.
    for(blas_size i = 0; i < nv; ++i) {
        std::copy(K + v_idx[i] * ldk, K + v_idx[i] * ldk + n, kv + i * n);
    }

    std::fill(g_sub, g_sub + n, 0.0);
    svm_compute_gsub_impl(n, nv, kv, K, ldk, y, s_idx, v_idx, g_sub, a_slack);
    for(auto i : s_idx) {
        g_sub[i] = -y[i];
    }

    double* ks = static_cast<double*>(blas_malloc(16, n * ns * sizeof(double)));
    double* k_io = static_cast<double*>(blas_malloc(16, n * (n + 1) / 2 * sizeof(double)));
    double* kchol = static_cast<double*>(blas_malloc(16, n * (n + 1) / 2 * sizeof(double)));

    for(blas_size i = 0; i < s_idx.size(); ++i) {
        std::copy(K + s_idx[i] * ldk, K + s_idx[i] * ldk + n, ks + i * n);
    }

    // compute cholesky decomposition of K into kchol (in RFP format)
    int info;
    dtrttf("N", "L", &n, K, &ldk, kchol, &info);
    dpftrf("N", "L", &n, kchol, &info);

    // kv now contains L_K^{-1} K_V
    dtfsm("N", "L", "L", "N", "N", &n, &nv, &one, kchol, kv, &n);

    // k_io contains KI = (K_v^T K^{-1} K_V)
    dsfrk("N", "L", "T", &nv, &n, &one, kv, &n, &zero, k_io);

    // k_io now contains its cholesky decomposition
    dpftrf("N", "L", &nv, k_io, &info);

    // kv now contains L_K^{-1} K_v L_I^{-1}^T
    dtfsm("N", "R", "L", "T", "N", &n, &nv, &one, k_io, kv, &n);

	// k_io now contains its inverse
	dpftri("N", "L", &nv, k_io, &info);

    // we make use of the fact that we have KI to compute a_slack for indices in V
    for(blas_size i = 0; i < nv; ++i) {
        a_slack[v_idx[i]] = 1 / (lambda * diagonal_element(nv, k_io, i, SymmetricFormat::RFP));
    }

    // we replace k_io with K^{-1/2}K_V KI^{-1} K_V K^{-1/2}
    dsfrk("N", "L", "N", &n, &nv, &one, kv, &n, &zero, k_io);

	// offset diagonal slightly for stability.
	offset_diagonal(n, k_io, 1e-15, false, SymmetricFormat::RFP);

    // compute the cholesky decomposition
    compute_cholesky(n, k_io, SymmetricFormat::RFP);

    // compute a for the indices in S
	dtfsm("N", "L", "L", "N", "N", &n, &ns, &one, kchol, ks, &n);

    for(blas_size i = 0; i < ns; ++i) {
        a_slack[s_idx[i]] = ddot(&n, ks, &ns, ks, &ns) / lambda;
    }

	triangular_multiply(MatrixTranspose::Transpose, n, ns, k_io, ks, n, SymmetricFormat::RFP);

    for(blas_size i = 0; i < ns; ++i) {
        a_slack[s_idx[i]] -= ddot(&n, ks, &ns, ks, &ns) / lambda;
    }

    blas_free(k_io);
    blas_free(kv);
    blas_free(ks);

    // a_slack now contains a * g
    std::transform(a_slack, a_slack + n, g_sub, a_slack, std::multiplies<>{});

    if(alo_predicted) {
        std::transform(a_slack, a_slack + n, y_hat, alo_predicted, std::plus<>{});
    }

    if(alo_mse) {
        double accumulator = 0;

        for(blas_size i = 0; i < n; ++i) {
			accumulator += std::max(0.0, 1 - y[i] * (y_hat[i] + a_slack[i]));
        }

        *alo_mse = accumulator / n;
    }

    blas_free(y_hat);
    blas_free(a_slack);
    blas_free(g_sub);
    blas_free(y_pred);
}