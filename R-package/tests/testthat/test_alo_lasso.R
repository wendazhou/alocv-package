context("ALO Lasso")
library(alocv)

compute_h_lasso_naive <- function(x, y, beta_hat, tol = 1e-5) {
    E <- abs(beta_hat) > tol
    W <- x[,E]
    K <- t(W) %*% W

    H <- W %*% solve(K, t(W))
    diag(H)
}

test_that("compute_h_lasso is equal to naive computation", {
    x <- matrix(rnorm(20 * 100), nrow = 100, ncol = 20)
    beta <- rnorm(20) * rbinom(20, 1, 0.5)
    y <- x %*% beta + rnorm(100, sd = 0.1)

    h_naive <- compute_h_lasso_naive(x, y, beta)
    h_fast <- compute_h_lasso(x, y, beta)

    expect_equal(h_fast, h_naive)
})

test_that("alo_lasso_rcpp is correct for one tuning", {
    x <- matrix(rnorm(20 * 100), nrow = 100, ncol = 20)
    beta <- matrix(rnorm(20) * rbinom(20, 1, 0.5), ncol=1)
    y <- x %*% beta + rnorm(100, sd = 0.1)

    h_naive <- compute_h_lasso_naive(x, y, beta)
    result_rcpp <- alo_lasso_rcpp(x, beta, y)

    expect_equal(h_naive, result_rcpp[['leverage']])
})

