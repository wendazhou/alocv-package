#include "catch.hpp"
#include <gram_utils.hpp>
#include <blas_configuration.h>

#include <algorithm>
#include <cmath>
#include <random>

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

bool increment_diag_correct(int n, bool skip_first) {
	auto init_values = make_random_matrices(n);

	auto X = init_values.first.get();
	auto X_rf = init_values.second.get();

	std::vector<double> initial_diagonal(n);

	for (int i = 0; i < n; ++i) {
		initial_diagonal[i] = X[i + n * i];
	}

    offset_diagonal(n, X_rf, 3.5, skip_first, SymmetricFormat::RFP);
	int info;
    dtfttr("N", "L", &n, X_rf, X, &n, &info);

    for(int i = skip_first ? 1 : 0; i < n; ++i) {
        if(X[i + n * i] != initial_diagonal[i] + 3.5) {
            return false;
        }
    }

    if(skip_first && X[0] != 0) {
        return false;
    }

    return true;
}

}

TEST_CASE("Offset Diagonal Correct for RFP (even)", "[RFP]") {
    REQUIRE(increment_diag_correct(6, false));
}

TEST_CASE("Offset Diagonal Correct for RFP (odd)", "[RFP]") {
    REQUIRE(increment_diag_correct(5, false));
}

TEST_CASE("Offset Diagonal with skip Correct for RFP (even)", "[RFP]") {
    REQUIRE(increment_diag_correct(6, true));
}

TEST_CASE("Offset Diagonal with skip correct for RFP (odd)", "[RFP]") {
    REQUIRE(increment_diag_correct(5, true));
}

namespace {

int test_diagonal_extract(int n) {
	auto init_values = make_random_matrices(n);

	auto X = init_values.first.get();
	auto X_rf = init_values.second.get();

	for (int i = 0; i < n; ++i) {
		auto diag = diagonal_element(n, X_rf, i, SymmetricFormat::RFP);

		if (diag != X[i + i * n]) {
			return i;
		}
	}

	return -1;
}

}

TEST_CASE("Extract Diagonal Element Correct for RFP (even)", "[RFP]") {
	REQUIRE(test_diagonal_extract(4) == -1);
}

TEST_CASE("Extract Diagonal Element Correct for RFP (odd)", "[RFP]") {
	REQUIRE(test_diagonal_extract(5) == -1);
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

std::pair<std::vector<double>, std::vector<double>> test_copy(int n, int k) {
	auto init_triangular = make_random_matrices(n, true);

	auto lhs_full = init_triangular.first.get();
	auto lhs_rfp = init_triangular.second.get();

	std::vector<double> my_copy(n);
	std::fill(my_copy.begin(), my_copy.end(), 0.0);

	copy_column(n, lhs_rfp, k, my_copy.data(), MatrixTranspose::Identity, SymmetricFormat::RFP);

	return std::make_pair(std::move(my_copy), std::vector<double>(lhs_full + k * n, lhs_full + k * n + n));
}

std::pair<std::vector<double>, std::vector<double>> test_copy_transpose(int n, int k) {
	auto init_triangular = make_random_matrices(n, true);

	auto lhs_full = init_triangular.first.get();
	auto lhs_rfp = init_triangular.second.get();

	std::vector<double> my_copy(n, 0.0);
	std::vector<double> reference(n, 0.0);

	copy_column(n, lhs_rfp, k, my_copy.data(), MatrixTranspose::Transpose, SymmetricFormat::RFP);
	strided_copy(lhs_full + k, lhs_full + k + n * n, reference.begin(), n, 1);

	return std::make_pair(std::move(my_copy), std::move(reference));
}

}


TEST_CASE("Copy Column Correct for RFP (even / first)", "[RFP]") {
	auto result = test_copy(4, 1);

	REQUIRE_THAT(result.first, Catch::Matchers::Equals(result.second));
}

TEST_CASE("Copy Column Correct for RFP (even / second)", "[RFP]") {
	auto result = test_copy(4, 3);

	REQUIRE_THAT(result.first, Catch::Matchers::Equals(result.second));
}

TEST_CASE("Copy Column Correct for RFP (odd / first)", "[RFP]") {
	auto result = test_copy(5, 1);

	REQUIRE_THAT(result.first, Catch::Matchers::Equals(result.second));
}

TEST_CASE("Copy Column Correct for RFP (odd / second)", "[RFP]") {
	auto result = test_copy(5, 3);

	REQUIRE_THAT(result.first, Catch::Matchers::Equals(result.second));
}

TEST_CASE("Copy Column Correct for RFP (even / first / transpose)", "[RFP]") {
	auto result = test_copy_transpose(4, 1);

	REQUIRE_THAT(result.first, Catch::Matchers::Equals(result.second));
}

TEST_CASE("Copy Column Correct for RFP (even / second / transpose)", "[RFP]") {
	auto result = test_copy_transpose(4, 3);

	REQUIRE_THAT(result.first, Catch::Matchers::Equals(result.second));
}

TEST_CASE("Copy Column Correct for RFP (odd / first / transpose)", "[RFP]") {
	auto result = test_copy_transpose(5, 1);

	REQUIRE_THAT(result.first, Catch::Matchers::Equals(result.second));
}

TEST_CASE("Copy Column Correct for RFP (odd / second / transpose)", "[RFP]") {
	auto result = test_copy_transpose(5, 3);

	REQUIRE_THAT(result.first, Catch::Matchers::Equals(result.second));
}
