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