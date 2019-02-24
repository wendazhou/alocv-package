#include "catch.hpp"

#include <alocv/alo_svm.h>
#include <blas_configuration.h>

#include "rfp_utils.h"

#include <algorithm>
#include <utility>
#include <vector>
#include <cmath>


TEST_CASE("ALO SVM Correct", "[SVM]") {
	// Includes a number of definitions
#include "examples/svm_example.in"

    double alo_hinge;
    double* leverage = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
    auto k_copy = blas_unique_alloc<double>(16, n * n);
    std::copy(K, K + n * n, k_copy.get());

    svc_compute_alo(n, k_copy.get(), y, alpha, rho, lambda, 1e-5, leverage, &alo_hinge);

    REQUIRE(alo_hinge == Approx(expected_hinge));

    blas_free(leverage);
}

TEST_CASE("ALO SVM Correct for RFP format", "[SVM]") {
#include "examples/svm_example.in"

    double alo_hinge;
    double* leverage = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
    auto K_rfp = blas_unique_alloc<double>(16, n * (n + 1) / 2);

	int info;
	dtrttf("N", "L", &n, K, &n, K_rfp.get(), &info);

	REQUIRE(info == 0);

	svc_compute_alo(n, K_rfp.get(), y, alpha, rho, lambda, 1e-5, leverage, &alo_hinge, true);

    REQUIRE(alo_hinge == Approx(expected_hinge));

    blas_free(leverage);
}

TEST_CASE("ALO SVM Correct For Triangular", "[SVM]") {
#include "examples/svm_example.in"

    double alo_hinge;
    double* leverage = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
    auto k_copy = blas_unique_alloc<double>(16, n * n);
	dlacpy("L", &n, &n, K, &n, k_copy.get(), &n);

    svc_compute_alo(n, k_copy.get(), y, alpha, rho, lambda, 1e-5, leverage, &alo_hinge, false, false);

    REQUIRE(alo_hinge == Approx(expected_hinge));

    blas_free(leverage);
}


TEST_CASE("ALO SVM Correct For Triangular with Pivoting", "[SVM]") {
#include "examples/svm_example.in"

    double alo_hinge;
    double* leverage = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
    auto k_copy = blas_unique_alloc<double>(16, n * n);
	dlacpy("L", &n, &n, K, &n, k_copy.get(), &n);

    svc_compute_alo(n, k_copy.get(), y, alpha, rho, lambda, 1e-5, leverage, &alo_hinge, false, true);

    REQUIRE(alo_hinge == Approx(expected_hinge));

    blas_free(leverage);
}

namespace {

std::pair<std::vector<double>, std::vector<double>> test_radial_kernel(blas_size n, blas_size p) {
    auto X = make_rectangular_matrix(n, p);

    std::vector<double> result_full(n * n, 0.0);
    std::vector<double> result_rfp(n * (n + 1) / 2, 0.0);
    std::vector<double> result_rfp_converted(n * n, 0.0);

    double gamma = 0.5;

    svm_kernel_radial(n, p, X.get(), gamma, result_full.data(), false);
    svm_kernel_radial(n, p, X.get(), gamma, result_rfp.data(), true);

    int info;
    dtfttr("N", "L", &n, result_rfp.data(), result_rfp_converted.data(), &n, &info);
    REQUIRE(info == 0);

    return std::make_pair(result_full, result_rfp_converted);
}}

TEST_CASE("ALO SVM Radial Kernel (even / fat)", "[SVM]") {
    auto result = test_radial_kernel(4, 5);
    REQUIRE_THAT(result.first, Catch::Matchers::Equals(result.second));
}

TEST_CASE("ALO SVM Radial Kernel (odd / fat)", "[SVM]") {
    auto result = test_radial_kernel(3, 5);
    REQUIRE_THAT(result.first, Catch::Matchers::Equals(result.second));
}


TEST_CASE("ALO SVM Radial Kernel (even / tall)", "[SVM]") {
    auto result = test_radial_kernel(10, 5);
    REQUIRE_THAT(result.first, Catch::Matchers::Equals(result.second));
}

TEST_CASE("ALO SVM Radial Kernel (odd / tall)", "[SVM]") {
    auto result = test_radial_kernel(9, 5);
    REQUIRE_THAT(result.first, Catch::Matchers::Equals(result.second));
}
