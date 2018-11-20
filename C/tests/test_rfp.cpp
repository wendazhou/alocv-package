#include "catch.hpp"
#include <gram_utils.hpp>
#include <blas_configuration.h>

#include <algorithm>
#include <cmath>
#include <random>


std::pair<unique_aligned_array<double>, unique_aligned_array<double>> make_random_matrices(int n, bool triangular=false) {
	auto X = blas_unique_alloc<double>(16, n * n);
	auto X_rf = blas_unique_alloc<double>(16, n * (n + 1) / 2);

    int info;

	for (int i = 0; i < n; ++i) {
		if (triangular) {
			std::fill(X.get() + i * n, X.get() + i * n + i, 0.0);
		}

		for (int j = triangular ? i : 0; j < n; ++j) {
			X[i * n + j] = std::sqrt(i) + std::sqrt(j);
		}
	}

    dtrttf("N", "L", &n, X.get(), &n, X_rf.get(), &info);

	return std::make_pair(std::move(X), std::move(X_rf));
}

namespace {

inline bool isclose(double a, double b) {
    return fabs(a - b) <= fmax(fabs(a), fabs(b)) * 1e-9;
}

void fill_random(double* data, int count) {
    std::mt19937 mt(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::generate(data, data + count, [&] {
        return dist(mt);
    });
}

bool approx_equal_lower(int n, double* A, double* B) {
    for(int i = 0; i < n; ++i) {
        for(int j = i; j <n; ++j) {
            double a = A[i * n + j];
            double b = B[i * n + j];

            if(!isclose(a, b)) {
                return false;
            }
        }
    }

    return true;
}

bool gram_matrix_correct(int n, int p) {
    double* X = (double*)blas_malloc(16, n * p * sizeof(double));
    double* L_rfp = (double*)blas_malloc(16, p * (p + 1) * sizeof(double) / 2);
    double* L_rfp_conv = (double*)blas_malloc(16, p * p * sizeof(double));
    double* L_full = (double*)blas_malloc(16, p * p * sizeof(double));

    fill_random(X, n * p);

    compute_gram(n, p, X, n, L_rfp, SymmetricFormat::RFP);
    compute_gram(n, p, X, n, L_full, SymmetricFormat::Full);

    int info;
    dtfttr("N", "L", &p, L_rfp, L_rfp_conv, &p, &info);

    bool result = approx_equal_lower(p, L_full, L_rfp_conv);

    blas_free(X);
    blas_free(L_rfp);
    blas_free(L_rfp_conv);
    blas_free(L_full);

    return result;
}

}

TEST_CASE("Gram Matrix Correct For RFP (even)", "[RFP]") {
    REQUIRE(gram_matrix_correct(6, 4));
}

TEST_CASE("Gram Matrix Correct For RFP (odd)", "[RFP]") {
    REQUIRE(gram_matrix_correct(6, 5));
}

namespace {

bool cholesky_correct(int n, int p) {
    double* X = (double*)blas_malloc(16, n * p * sizeof(double));
    double* L_rfp = (double*)blas_malloc(16, p * (p + 1) * sizeof(double) / 2);
    double* L_rfp_conv = (double*)blas_malloc(16, p * p * sizeof(double));
    double* L_full = (double*)blas_malloc(16, p * p * sizeof(double));

    fill_random(X, n * p);

    compute_gram(n, p, X, n, L_rfp, SymmetricFormat::RFP);
    compute_gram(n, p, X, n, L_full, SymmetricFormat::Full);

    compute_cholesky(p, L_rfp, SymmetricFormat::RFP);
    compute_cholesky(p, L_full, SymmetricFormat::Full);

    int info;
    dtfttr("N", "L", &p, L_rfp, L_rfp_conv, &p, &info);

    bool result = approx_equal_lower(p, L_full, L_rfp_conv);

    blas_free(X);
    blas_free(L_rfp);
    blas_free(L_rfp_conv);
    blas_free(L_full);

    return result;
}

}

TEST_CASE("Cholesky Correct for RFP (even)", "[RFP]") {
    REQUIRE(cholesky_correct(6, 4));
}

TEST_CASE("Cholesky Correct for RFP (odd)", "[RFP]") {
    REQUIRE(cholesky_correct(6, 5));
}

namespace {

bool inverse_correct(int n, int p) {
    double* X_1 = (double*)blas_malloc(16, n * p * sizeof(double));
    double* X_2 = (double*)blas_malloc(16, n * p * sizeof(double));
    double* L_rfp = (double*)blas_malloc(16, p * (p + 1) * sizeof(double) / 2);
    double* L_rfp_conv = (double*)blas_malloc(16, p * p * sizeof(double));

    fill_random(X_1, n * p);
    std::copy(X_1, X_1 + n * p, X_2);
    fill_random(L_rfp, p * (p + 1) / 2);

    int info;
    dtfttr("N", "L", &p, L_rfp, L_rfp_conv, &p, &info);

    solve_triangular(n, p, L_rfp, X_1, n, SymmetricFormat::RFP);
    solve_triangular(n, p, L_rfp_conv, X_2, n, SymmetricFormat::Full);

    return std::equal(X_1, X_1 + n * p, X_2, X_2 + n * p, &isclose);
}

}

TEST_CASE("Triangular Inverse Correct for RFP (even)", "[RFP]") {
    REQUIRE(inverse_correct(6, 4));
}

TEST_CASE("Triangular Inverse Correct for RFP (odd)", "[RFP]") {
    REQUIRE(inverse_correct(6, 5));
}


namespace {

unique_aligned_array<double> make_rectangular_matrix(int n, int m) {
	auto X = blas_unique_alloc<double>(16, n * m);
	std::generate(X.get(), X.get() + n * m, [counter = 0]() mutable { return counter++;  });

	return X;
}

std::pair<unique_aligned_array<double>, unique_aligned_array<double>> compute_triangular_multiplication(int n, int m, bool trans) {
	auto init_triangular = make_random_matrices(n);
	auto rhs = make_rectangular_matrix(n, m);
	auto rhs2 = make_rectangular_matrix(n, m);

	auto lhs_full = init_triangular.first.get();
	auto lhs_rfp = init_triangular.second.get();

	auto transa = trans ? MatrixTranspose::Transpose : MatrixTranspose::Identity;

	triangular_multiply(transa, n, m, lhs_full, rhs.get(), n, SymmetricFormat::Full);
	triangular_multiply(transa, n, m, lhs_rfp, rhs2.get(), n, SymmetricFormat::RFP);

	return std::make_pair(std::move(rhs), std::move(rhs2));
}

}

TEST_CASE("Triangular Matrix Multiply Correct for RFP (even)", "[RFP]") {
	auto results = compute_triangular_multiplication(6, 4, false);

	for (int i = 0; i < 24; ++i) {
		REQUIRE(results.first[i] == Approx(results.second[i]));
	}
}


TEST_CASE("Triangular Matrix Multiply Correct for RFP (odd)", "[RFP]") {
	auto results = compute_triangular_multiplication(5, 4, false);

	for (int i = 0; i < 20; ++i) {
		REQUIRE(results.first[i] == Approx(results.second[i]));
	}
}


TEST_CASE("Triangular Matrix Multiply Correct for RFP (even / T)", "[RFP]") {
	auto results = compute_triangular_multiplication(6, 4, true);

	for (int i = 0; i < 24; ++i) {
		REQUIRE(results.first[i] == Approx(results.second[i]));
	}
}


TEST_CASE("Triangular Matrix Multiply Correct for RFP (odd / T)", "[RFP]") {
	auto results = compute_triangular_multiplication(5, 4, true);

	for (int i = 0; i < 20; ++i) {
		REQUIRE(results.first[i] == Approx(results.second[i]));
	}
}


namespace {

std::pair<unique_aligned_array<double>, unique_aligned_array<double>> compute_symmetric_multiplication(int n, int m) {
	auto init_triangular = make_random_matrices(n);
	auto rhs = make_rectangular_matrix(n, m);
	auto rhs2 = make_rectangular_matrix(n, m);

	auto rhs_ptr = rhs.get();
	auto rhs2_ptr = rhs2.get();

	auto lhs_full = init_triangular.first.get();
	auto lhs_rfp = init_triangular.second.get();

	symmetric_multiply(n, m, lhs_full, rhs_ptr, n, SymmetricFormat::Full);
	symmetric_multiply(n, m, lhs_rfp, rhs2_ptr, n, SymmetricFormat::RFP);

	return std::make_pair(std::move(rhs), std::move(rhs2));
}}

TEST_CASE("Symmetric Matrix Multiply Correct for RFP (odd)", "[RFP]") {
	auto results = compute_symmetric_multiplication(5, 4);

	for (int i = 0; i < 20; ++i) {
		REQUIRE(results.first[i] == Approx(results.second[i]));
	}
}


TEST_CASE("Symmetric Matrix Multiply Correct for RFP (even)", "[RFP]") {
	auto results = compute_symmetric_multiplication(4, 3);

	for (int i = 0; i < 12; ++i) {
		REQUIRE(results.first[i] == Approx(results.second[i]));
	}
}


TEST_CASE("Symmetric Matrix Multiply Correct for Full", "[RFP]") {
	blas_size m = 4;
	blas_size n = 3;
	blas_size ldb = 6;

	auto init = make_random_matrices(m);
	auto rhs = make_rectangular_matrix(ldb, 3);

	symmetric_multiply(4, 3, init.first.get(), rhs.get(), 6, SymmetricFormat::Full);

	std::vector<double> result(4 * 3);
	dlacpy("N", &m, &n, rhs.get(), &ldb, result.data(), &m);

	std::vector<double> expected = {
		9.02457955,  15.02457955,  17.50986092,  19.41688439,
		33.90216577,  63.90216577,  76.32857264,  85.86368999,
		58.77975199, 112.77975199, 135.14728435, 152.3104956 };

	for (blas_size i = 0; i < m * n; ++i) {
		REQUIRE(result[i] == Approx(expected[i]));
	}
}


TEST_CASE("Symmetric Matrix Multiply Correct for Full Stride A", "[RFP][!shouldfail]") {
	blas_size m1 = 3;
	blas_size m2 = 2;
	blas_size n = 4;
	blas_size ldb = 5;
	blas_size lda = 5;

	auto init = make_random_matrices(lda);
	auto rhs_storage = make_rectangular_matrix(ldb, n);

	double* a = init.first.get() + m1 + m1 * lda;
	double* a_rfp = init.second.get() + lda;
	double* rhs_ptr = rhs_storage.get() + m1;

	double one = 1.0;
	double zero = 0.0;

	dsymm("L", "L", &m2, &n, &one, a, &lda, rhs_ptr, &ldb, &zero, rhs_ptr, &ldb);

	std::vector<double> expected = {
		25.32050808,  27.19615242,  61.30127019,  65.85640646,
		97.2820323 , 104.5166605 , 133.26279442, 143.17691454
	};

	std::vector<double> result(expected.size());
	dlacpy("N", &m2, &n, rhs_ptr, &ldb, result.data(), &m2);

	for (blas_size i = 0; i < m2 * n; ++i) {
		REQUIRE(result[i] == Approx(expected[i]));
	}
}
