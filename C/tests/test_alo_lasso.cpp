#include <alocv/alo_lasso.h>
#include "catch.hpp"

TEST_CASE("ALO is correct for LASSO") {
#include "examples/lasso_example.in"
    std::vector<double> alo(num_tuning);

    lasso_compute_alo_d(n, p, num_tuning, x, n, beta, p, y, 1, 1e-5, alo.data(), nullptr);

    for (size_t i = 0; i < alo.size(); ++i) {
        REQUIRE(alo[i] == Approx(alo_expected[i]));
    }
}
