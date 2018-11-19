#include <alocv/alo_svm.h>
#include "blas_configuration.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include <vector>

#include "gram_utils.hpp"


namespace {

void svm_compute_gsub_impl(blas_size n, blas_size nv, double* kv,
                           const double* K, blas_size ldk, const double* y,
                           const std::vector<blas_size>& s_idx, const std::vector<blas_size>& v_idx,
						   double* g_sub, double* g_temp) {
    blas_size one_i = 1;

    double temp_size;
    blas_size info;
    blas_size min_one_i = -1;

    // before we do anything, we can compute the g vector
    // we start by computing the entries in V
    for(auto i : s_idx) {
        daxpy(&n, y + i, K + i * ldk, &one_i, g_temp, &one_i);
    }

    dgels("N", &n, &nv, &one_i, kv, &n, g_temp, &n, &temp_size, &min_one_i, &info);

    blas_size work_size = static_cast<blas_size>(temp_size);
    double* work = static_cast<double*>(blas_malloc(16, work_size * sizeof(double)));

    dgels("N", &n, &nv, &one_i, kv, &n, g_temp, &n, work, &work_size, &info);

    blas_free(work);

    for(blas_size i = 0; i < v_idx.size(); ++i) {
        g_sub[v_idx[i]] = g_temp[i];
    }
}


void svm_compute_a_impl(blas_size n, blas_size nv, blas_size ns, double* kkv, double* kks,
						const std::vector<blas_size>& v_idx, const std::vector<blas_size>& s_idx,
	                    double* a_slack, double lambda) {
	auto tau_storage = blas_unique_alloc<double>(16, nv);
	auto tau = tau_storage.get();

	blas_size min_one_i = -1;
	blas_size lwork = -1;
	blas_size info;
	double work_size;

	dgeqrf(&n, &nv, kkv, &n, tau, &work_size, &min_one_i, &info);
	lwork = static_cast<blas_size>(work_size);
	auto work_storage = blas_unique_alloc<double>(16, lwork);
	auto work = work_storage.get();

	dgeqrf(&n, &nv, kkv, &n, tau, work, &lwork, &info);

	{
		blas_size one_i = 1;
		for(blas_size i = 0; i < ns; ++i) {
			a_slack[s_idx[i]] = ddot(&n, kks + i * n, &one_i, kks + i * n, &one_i) / lambda;
		}

		dormqr("L", "T", &n, &ns, &nv, kkv, &n, tau, kks, &n, work, &lwork, &info);

		for(blas_size i = 0; i < ns; ++i) {
			a_slack[s_idx[i]] -= ddot(&nv, kks + i * n, &one_i, kks + i * n, &one_i) / lambda;
		}
	}

	{
		dtrtri("U", "N", &nv, kkv, &n, &info);

		for (blas_size i = 0; i < nv; ++i) {
			blas_size num_elements = nv - i;
			auto start = kkv + i + i * n;
			a_slack[v_idx[i]] = 1 / (lambda * ddot(&num_elements, start, &n, start, &n));
		}
	}
}

}


void svm_compute_alo(blas_size n, double* K, const double* y, const double* alpha,
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
    dsymv("L", &n, &one, K, &n, alpha, &one_i, &zero, y_pred, &one_i);

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
        std::copy(K + v_idx[i] * n, K + v_idx[i] * n + n, kv + i * n);
    }

    std::fill(g_sub, g_sub + n, 0.0);
    svm_compute_gsub_impl(n, nv, kv, K, n, y, s_idx, v_idx, g_sub, a_slack);
    for(auto i : s_idx) {
        g_sub[i] = -y[i];
    }

    double* ks = static_cast<double*>(blas_malloc(16, n * ns * sizeof(double)));
    double* kchol = static_cast<double*>(blas_malloc(16, n * (n + 1) / 2 * sizeof(double)));

    for(blas_size i = 0; i < s_idx.size(); ++i) {
        std::copy(K + s_idx[i] * n, K + s_idx[i] * n + n, ks + i * n);
    }

    // compute cholesky decomposition of K
	compute_cholesky(n, K, SymmetricFormat::Full);

    // kv now contains L_K^{-1} K_V = L_K^T[,V]
	std::fill(kv, kv + nv * n, 0.0);
	for (blas_size i = 0; i < nv; ++i) {
		copy_column(n, K, v_idx[i], kv + i * n, MatrixTranspose::Transpose, SymmetricFormat::Full);
	}

	// similarly, ks now contains L_K^{-1} K_S = L_K^T[,S]
	std::fill(ks, ks + ns * n, 0.0);
	for (blas_size i = 0; i < ns; ++i) {
		copy_column(n, K, s_idx[i], ks + i * n, MatrixTranspose::Transpose, SymmetricFormat::Full);
	}

	svm_compute_a_impl(n, nv, ns, kv, ks, v_idx, s_idx, a_slack, lambda);

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