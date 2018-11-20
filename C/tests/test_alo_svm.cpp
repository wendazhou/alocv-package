#include "catch.hpp"

#include <alocv/alo_svm.h>
#include <blas_configuration.h>

#include <algorithm>
#include <cmath>


TEST_CASE("ALO SVM Correct", "[SVM]") {
	// Includes a number of definitions
#include "examples/svm_example.in"

    double alo_hinge;
    double* leverage = static_cast<double*>(blas_malloc(16, n * sizeof(double)));
    auto k_copy = blas_unique_alloc<double>(16, n * n);
    std::copy(K, K + n * n, k_copy.get());

    svm_compute_alo(n, k_copy.get(), y, alpha, rho, lambda, 1e-5, leverage, &alo_hinge);

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

	svm_compute_alo(n, K_rfp.get(), y, alpha, rho, lambda, 1e-5, leverage, &alo_hinge, true);

    REQUIRE(alo_hinge == Approx(expected_hinge));

    blas_free(leverage);
}