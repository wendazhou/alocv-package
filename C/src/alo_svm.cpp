#include <alocv/alo_svm.h>
#include "blas_configuration.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <functional>
#include <vector>

#include "gram_utils.hpp"


namespace {


void svm_compute_gsub_impl(blas_size n, blas_size nv, double* kv,
                           const double* K, blas_size ldk, const double* y,
                           const std::vector<blas_size>& s_idx, const std::vector<blas_size>& v_idx,
						   double* g_sub, double* g_temp, SymmetricFormat format) {
    blas_size one_i = 1;

    blas_size info;
    blas_size min_one_i = -1;

    // before we do anything, we can compute the g vector
    // we start by computing the entries in V
    for(auto i : s_idx) {
        copy_add_column(n, K, i, y[i], g_temp, format);
    }

    blas_size work_size;

    {
        // workspace query for GELS
        double temp_size;
        dgels("N", &n, &nv, &one_i, kv, &n, g_temp, &n, &temp_size, &min_one_i, &info);
        work_size = static_cast<blas_size>(temp_size);
    }

    double* work = static_cast<double*>(blas_malloc(16, work_size * sizeof(double)));

    dgels("N", &n, &nv, &one_i, kv, &n, g_temp, &n, work, &work_size, &info);

    blas_free(work);

    for(blas_size i = 0; i < nv; ++i) {
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

	{
		// work size query
		double work_size;
		dgeqrf(&n, &nv, kkv, &n, tau, &work_size, &min_one_i, &info);
		lwork = static_cast<blas_size>(work_size);
		dormqr("L", "T", &n, &ns, &nv, kkv, &n, tau, kks, &n, &work_size, &min_one_i, &info);
		lwork = std::max(lwork, static_cast<blas_size>(work_size));
	}

	auto work_storage = blas_unique_alloc<double>(16, lwork);
	auto work = work_storage.get();

	dgeqrf(&n, &nv, kkv, &n, tau, work, &lwork, &info);

	{
        // compute a_slack for the elements in S
        // For these elements, a_slack is given as a difference of two quadratic forms.
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
        // compute a_slack for the elements in V
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
                     double rho, double lambda, double tol, double* alo_predicted, double* alo_hinge,
					 bool use_rfp, bool use_pivoting) {
	SymmetricFormat format = use_rfp ? SymmetricFormat::RFP : SymmetricFormat::Full;
    double* y_pred = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
    double* y_hat = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
    double* a_slack = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
    double* g_sub = static_cast<double*>(blas_malloc(16, n * sizeof(double)));

    std::unique_ptr<blas_size[]> pivot = nullptr;

    if (use_pivoting) {
        pivot.reset(new blas_size[n]);
    }

    // we compute K * alpha in y_pred
	symmetric_multiply(n, 1, K, alpha, n, y_pred, n, format);

    // a_slack will contain -lambda * y_hat
    std::transform(y_pred, y_pred + n, a_slack, [=](double x) { return -x * lambda; });

	// Add offset so that y_pred now contains y_hat
	std::transform(y_pred, y_pred + n, y_pred, [=](double x) { return x - rho; });

    // save the predicted values in y_hat
	std::copy(y_pred, y_pred + n, y_hat);

    // y_pred will contain y * y_hat
    std::transform(y_pred, y_pred + n, y, y_pred, std::multiplies<double>{});

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
		copy_column(n, K, v_idx[i], kv + i * n, MatrixTranspose::Identity, format, true);
    }

    std::fill(g_sub, g_sub + n, 0.0);
    svm_compute_gsub_impl(n, nv, kv, K, n, y, s_idx, v_idx, g_sub, a_slack, format);
    for(auto i : s_idx) {
        g_sub[i] = -y[i];
    }

    double* ks = static_cast<double*>(blas_malloc(16, n * ns * sizeof(double)));

    // compute cholesky decomposition of K
    if (format == SymmetricFormat::Full && use_pivoting) {
        blas_size rank;
        blas_size info;
        auto work = blas_unique_alloc<double>(16, 2 * n);
        auto chol_pivot = blas_unique_alloc<blas_size>(16, n);

        dpstrf("L", &n, K, &n, chol_pivot.get(), &rank, &tol, work.get(), &info);

        for (blas_size i = 0; i < n; ++i) {
            pivot[i] = i;
        }

        std::sort(pivot.get(), pivot.get() + n, [&](blas_size i1, blas_size i2) {
            return chol_pivot[i1] < chol_pivot[i2];
        });
    }
    else {
	    compute_cholesky(n, K, format);
    }

    // kv now contains L_K^{-1} K_V = L_K^T[,V]
	std::fill(kv, kv + nv * n, 0.0);
	for (blas_size i = 0; i < nv; ++i) {
        auto target_column = pivot ? pivot[v_idx[i]] : v_idx[i];
		copy_column(n, K, target_column, kv + i * n, MatrixTranspose::Transpose, format);
	}

	// similarly, ks now contains L_K^{-1} K_S = L_K^T[,S]
	std::fill(ks, ks + ns * n, 0.0);
	for (blas_size i = 0; i < ns; ++i) {
        auto target_column = pivot ? pivot[s_idx[i]] : s_idx[i];
		copy_column(n, K, target_column, ks + i * n, MatrixTranspose::Transpose, format);
	}

	svm_compute_a_impl(n, nv, ns, kv, ks, v_idx, s_idx, a_slack, lambda);

    blas_free(kv);
    blas_free(ks);

    // a_slack now contains a * g
    std::transform(a_slack, a_slack + n, g_sub, a_slack, std::multiplies<double>{});

    if(alo_predicted) {
        std::transform(a_slack, a_slack + n, y_hat, alo_predicted, std::plus<double>{});
    }

    if(alo_hinge) {
        double accumulator = 0;

        for(blas_size i = 0; i < n; ++i) {
			accumulator += std::max(0.0, 1 - y[i] * (y_hat[i] + a_slack[i]));
        }

        *alo_hinge = accumulator / n;
    }

    blas_free(y_hat);
    blas_free(a_slack);
    blas_free(g_sub);
    blas_free(y_pred);
}


void svm_kernel_radial(blas_size n, blas_size p, const double* X, double gamma, double* K, bool use_rfp) {
	for (blas_size i = 0; i < n; ++i) {
		for (blas_size j = 0; j <= i; ++j) {
			double value = 0.0;

			for (blas_size k = 0; k < p; ++k) {
				double x_v = X[i + k * n];
				double y_v = X[j + k * n];
				value += (x_v - y_v) * (x_v - y_v);
			}

			value = std::exp(-gamma * value);

			if (use_rfp) {
				*index_rfp(n, K, i, j) = value;
			}
			else {
				K[i + j * n] = value;
			}
		}
	}
}


void svm_kernel_polynomial(blas_size n, blas_size p, const double* X, double* K, double gamma, double degree, double coef0, bool use_rfp) {
    compute_gram(n, p, X, n, K, MatrixTranspose::Transpose, use_rfp ? SymmetricFormat::RFP : SymmetricFormat::Full);

    if (use_rfp) {
        std::transform(K, K + n * (n + 1) / 2, K, [=](double x) {
            return std::pow(gamma * x + coef0, degree);
        });
    }
    else {
        for (blas_size j = 0; j < n; ++j) {
            for (blas_size i = j; i < n; ++i) {
                K[i + j * n] = std::pow(gamma * K[i + j * n] + coef0, degree);
            }
        }
    }
}
