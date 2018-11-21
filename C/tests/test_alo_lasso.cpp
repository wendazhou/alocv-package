#include <alocv/alo_lasso.h>
#include "catch.hpp"

TEST_CASE("ALO is correct for LASSO", "[LASSO]") {
#include "examples/lasso_example.in"
    std::vector<double> alo(num_tuning);
    std::vector<double> leverage(num_tuning * n);

    lasso_compute_alo_d(n, p, num_tuning, x, n, beta, p, y, 1, 1e-5, alo.data(), leverage.data());

    for (size_t i = 0; i < alo.size(); ++i) {
        REQUIRE(alo[i] == Approx(alo_expected[i]));
    }
}


TEST_CASE("ALO is correct for LASSO (no leverage)", "[LASSO]") {
#include "examples/lasso_example.in"
    std::vector<double> alo(num_tuning);
    std::vector<double> leverage(num_tuning * n);

    lasso_compute_alo_d(n, p, num_tuning, x, n, beta, p, y, 1, 1e-5, alo.data(), nullptr);

    for (size_t i = 0; i < alo.size(); ++i) {
        REQUIRE(alo[i] == Approx(alo_expected[i]));
    }
}
