#include "alocv/alo_enet.h"
#include "blas_configuration.h"
#include "lasso_utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>

#include "gram_utils.hpp"

namespace {

blas_size sym_num_elements(blas_size p, SymmetricFormat format) {
    if(format == SymmetricFormat::Full) {
        return p * p;
    }
    else {
        return p * (p + 1) / 2;
    }
}

/** Compute the ALO leverage for the elastic net.
 *
 * @param n The number of observations
 * @param p The number of active parameters
 * @param[in, out] XE A n x p matrix containing the active set
 * @param lde The leading dimension of E
 * @param lambda The value of the regularizer lambda
 * @param alpha The value of the elastic net parameter alpha
 * @param has_intercept Whether an intercept was fit to the data
 * @param[out] h A vector of length n containing the leverage value for each observation.
 * @param[out] L If provided, a temporary array of size at least p * p to store the inner products.
 * @param format The format to use for storing intermediate products.
 *
 */
void alo_elastic_net_rfp(blas_size n, blas_size p, double* XE, blas_size lde,
                         double lambda, double alpha, bool has_intercept,
                         double* h, double* L, SymmetricFormat format) {
    bool alloc_l = false;
    blas_size p_effective = p + (has_intercept ? 1 : 0);

    if (!L) {
        L = (double*)blas_malloc(16, sizeof(double) * sym_num_elements(p_effective, format));
    }

    compute_gram(n, p_effective, XE, lde, L, format);

    if (alpha != 1) {
        double offset = (1 - alpha) * lambda;
        offset_diagonal(p_effective, L, offset, has_intercept, format);
    }

    compute_cholesky(p_effective, L, format);
    solve_triangular(n, p_effective, L, XE, lde, format);

    for(blas_size i = 0; i < n; ++i) {
        h[i] = ddot(&p_effective, XE + i, &n, XE + i, &n);
    }

    if (alloc_l) {
        blas_free(L);
    }
}


void sqrt_second_derivative_logit(blas_size n, const double* y_fitted, double* output) {
    std::transform(y_fitted, y_fitted + n, output, [](double x) { return 0.5 / cosh(x / 2); });
}

void sqrt_second_derivative_log(blas_size n, const double* y_fitted, double* output) {
    std::transform(y_fitted, y_fitted + n, output, [](double x) { return exp(x / 2); });
}


/*! Scales the rows of the copied predictors by the specified amount.
 *
 * @param n The number of rows in XE
 * @param p The number of columns in
 * @param[in, out] XE The matrix of predictors to scale
 * @param lde The leading dimension of XE
 * @param[in] scale A vector of length n containing the scaling for each row
 *
 */
void scale_by_row(blas_size n, blas_size p, double* __restrict XE, blas_size lde, const double* __restrict scale) {
    for(int i = 0; i < p; ++i) {
        for(int j = 0; j < n; ++j) {
            XE[j + lde * i] *= scale[j];
        }
    }
}


/*! Scales the predictor matrix for GLM models.
 *
 * @param n The number of observations.
 * @param p The number of active predictors.
 * @param[in, out] XE The matrix containing the predictors.
 * @param lde The leading dimension of XE.
 * @param y_fitted The computed link-space responses.
 * @param family The GLM family currently being fit.
 *
 */
void scale_predictors_glm(blas_size n, blas_size p, double* XE, blas_size lde, double const* y_fitted, GlmFamily family) {
    if(family == GlmFamily::GlmFamilyGaussian) {
        return;
    }

    double* scale = (double*)blas_malloc(16, n * sizeof(double));

    switch(family) {
    case GlmFamily::GlmFamilyLogit:
        sqrt_second_derivative_logit(n, y_fitted, scale);
        break;
    case GlmFamily::GlmFamilyPoisson:
        sqrt_second_derivative_log(n, y_fitted, scale);
        break;
    default:
        return;
    }

    scale_by_row(n, p, XE, lde, scale);

    blas_free(scale);
}

struct AloResult {
    double deviance;
    double mse;
    double mae;
};

template<typename Config>
AloResult compute_alo_fitted_impl(blas_size n, const double* y, const double* y_fitted, const double* leverage) {
    typename Config::Grad grad;
    typename Config::Grad2 grad2;
    typename Config::LossDeviance deviance;
    typename Config::LossMse mse;
    typename Config::LossMae mae;

    double acc_dev = 0;
    double acc_mse = 0;
    double acc_mae = 0;

    for(blas_size i = 0; i < n; ++i) {
        double y_alo = y_fitted[i] + leverage[i] * grad(y[i], y_fitted[i]) / grad2(y[i], y_fitted[i]) / (1 - leverage[i]);
        acc_dev += deviance(y[i], y_alo);
        acc_mse += mse(y[i], y_alo);
        acc_mae += mae(y[i], y_alo);
    }

    AloResult result;

    result.deviance = acc_dev / n;
    result.mse = acc_mse / n;
    result.mae = acc_mae / n;

    return result;
}

struct SquareError {
    double operator()(double x) {
        return x * x;
    }
};

struct AbsoluteError {
    double operator()(double x) {
        return abs(x);
    }
};

struct GaussianGlmConfig {
    struct Grad {
        double operator()(double y, double y_fitted) const {
            return y_fitted - y;
        }
    };

    struct Grad2 {
        double operator()(double y, double y_fitted) const {
            return 1;
        }
    };

    template<typename Metric>
    struct LossMetric {
        double operator()(double y, double y_fitted) const {
            Metric metric;
            return metric(y - y_fitted);
        }
    };

    typedef LossMetric<SquareError> LossDeviance;
    typedef LossMetric<SquareError> LossMse;
    typedef LossMetric<AbsoluteError> LossMae;
};

struct LogisticGlmConfig {
    struct Grad {
        double operator()(double y, double y_fitted) const {
            return 1 / (1 + exp(-y_fitted)) - y;
        }
    };

    struct Grad2 {
        double operator()(double y, double y_fitted) const {
            double cx = cosh(y_fitted / 2);
            return 0.25 / (cx * cx);
        }
    };

    struct LossDeviance {
        double operator()(double y, double y_fitted) const {
            y_fitted = std::max(std::min(y_fitted, 11.5), -11.5);

            double lp = y * log1p(exp(-y_fitted)) + (1 - y) * log1p(exp(y_fitted));
            return 2 * lp;
        }
    };

    template<typename Metric>
    struct LossMetric {
        double operator()(double y, double y_fitted) const {
            Metric metric;

            double mu_fitted  = 1 / (1 + exp(-y_fitted));
            double res = (y - mu_fitted);
            return 2 * metric(res);
        }
    };

    typedef LossMetric<SquareError> LossMse;
    typedef LossMetric<AbsoluteError> LossMae;
};

struct PoissonGlmConfig {
    struct Grad {
        double operator()(double y, double y_fitted) const {
            return exp(y_fitted) - y;
        }
    };

    struct Grad2 {
        double operator()(double y, double y_fitted) const {
            return exp(y_fitted);
        }
    };

    struct LossDeviance {
        double operator()(double y, double y_fitted) const {
            if(y > 0) {
                return y * log(y) - y + exp(y_fitted) - y * y_fitted;
            } else {
                return exp(y_fitted);
            }
        }
    };

    template<typename Metric>
    struct LossMetric {
        double operator()(double y, double y_fitted) const {
            Metric metric;
            double mu_fitted = exp(y_fitted);

            return metric(y - mu_fitted);
        }
    };

    typedef LossMetric<SquareError> LossMse;
    typedef LossMetric<AbsoluteError> LossMae;
};


AloResult compute_alo_fitted_glm(blas_size n, const double* y, const double* y_fitted, const double* leverage, GlmFamily family) {
    switch(family) {
    case GlmFamily::GlmFamilyGaussian:
        return compute_alo_fitted_impl<GaussianGlmConfig>(n, y, y_fitted, leverage);
    case GlmFamily::GlmFamilyLogit:
        return compute_alo_fitted_impl<LogisticGlmConfig>(n, y, y_fitted, leverage);
    case GlmFamily::GlmFamilyPoisson:
        return compute_alo_fitted_impl<PoissonGlmConfig>(n, y, y_fitted, leverage);
    default:
        return AloResult{ -1.0, -1.0, -1.0 };
    }
}

}


void compute_fitted(blas_size n, blas_size k, const double* XE,
                    const double* beta, double a0, bool has_intercept,
                    const std::vector<blas_size>& index, double* y_fitted) {
	auto beta_active_storage = blas_unique_alloc<double>(16, static_cast<blas_size>(index.size()) + has_intercept);
	double* beta_active = beta_active_storage.get();

    if(has_intercept) {
        beta_active[0] = a0;
        k += 1;
    }

    for(int i = 0; i < index.size(); ++i) {
        beta_active[i + has_intercept] = beta[index[i]];
    }

    double zero = 0.0;
    double one = 1.0;
    blas_size one_i = 1;

    dgemv("N", &n, &k, &one, XE, &n, beta_active, &one_i, &zero, y_fitted, &one_i);
}


double compute_alo_fitted(blas_size n, const double* y, const double* y_fitted, const double* leverage) {
    double acc = 0;

    for(blas_size i = 0; i < n; ++i) {
        double res = (y[i] - y_fitted[i]) / (1 - leverage[i]);
        acc += res * res;
    }

    return acc / n;
}


void enet_compute_alo_d(blas_size n, blas_size p, blas_size m, const double* A, blas_size lda,
                        const double* B, blas_size ldb, const double* y, const double* a0,
                        const double* lambda, double alpha,
                        int has_intercept, GlmFamily family, int use_rfp,
                        double tolerance, double* alo, double* leverage,
                        double* alo_mse, double* alo_mae) {
    SymmetricFormat format = use_rfp ? SymmetricFormat::RFP : SymmetricFormat::Full;

    blas_size max_active = max_active_set_size(m, p, B, ldb, tolerance) + (has_intercept ? 1 : 0);
    const blas_size l_size = sym_num_elements(max_active, format);

    double* y_fitted = (double*)blas_malloc(16, n * sizeof(double));
    double* L = (double*)blas_malloc(16, l_size * sizeof(double));
    double* XE = (double*)blas_malloc(16, max_active * n * sizeof(double));

    blas_size ld_leverage;
    bool alloc_leverage;
    if (leverage) {
        alloc_leverage = false;
        ld_leverage = n;
    }
    else {
        alloc_leverage = true;
        leverage = (double*)blas_malloc(16, n * sizeof(double));
        ld_leverage = 0;
    }

    for(blas_size i = 0; i < m; ++i) {
        std::vector<blas_size> current_index = find_active_set(p, B + ldb * i, tolerance);

        if(current_index.empty() && !has_intercept) {
            // no selected variables
            std::fill(leverage + i * ld_leverage, leverage + i * ld_leverage + n, 0.0);
            std::fill(y_fitted, y_fitted + n, 0.0);
        } else {
            copy_active_set(n, A, lda, has_intercept, current_index, std::vector<blas_size>(), XE, n);
            compute_fitted(n, static_cast<blas_size>(current_index.size()),
						   XE, B + ldb * i, has_intercept ? a0[i] : 0.0,
                           has_intercept, current_index, y_fitted);

            scale_predictors_glm(n, static_cast<blas_size>(current_index.size()), XE, n, y_fitted, family);

            alo_elastic_net_rfp(
                n, static_cast<blas_size>(current_index.size()), XE, n, lambda[i], alpha, has_intercept,
                leverage + i * ld_leverage, L, format);
        }

        AloResult result = compute_alo_fitted_glm(n, y, y_fitted, leverage + i * ld_leverage, family);
        alo[i] = result.deviance;

        if(alo_mse) {
            alo_mse[i] = result.mse;
        }

        if(alo_mae) {
            alo_mae[i] = result.mae;
        }
    }

    if(alloc_leverage) {
        blas_free(leverage);
    }

    blas_free(y_fitted);
    blas_free(L);
    blas_free(XE);
}
