context("ALO SVM")
library(alocv)
library(e1071)


make_example <- function(n, p, seed=42) {
    n2 <- round(n / 2)
    X <- withr::with_seed(seed, matrix(rnorm(n / 2 * p), nrow=n, ncol=p))
    X[1:n2,] <- X[1:n2,] + 2.5
    X[(n2+1):n,] <- X[(n2+1):n,] - 2.5
    y <- c(rep(1, n/2), rep(-1, n/2 - 1), 1)
    list(X = X, y = y)
}


test_that("ALO SVM Correct for Gaussian RBF", {
    df <- make_example(200, 50)
    g <- 2 / 50

    svm.fitalo <- alocv::alo_svm(df$X, df$y, scale = F, kernel='radial', gamma=g, cost=1,
                                 type='C-classification', tolerance=1e-5, use_rfp=FALSE)

    expect_equal(svm.fitalo$alo_loss, 0.346218829410011)
})


test_that("ALO SVM Correct for Gaussian RBF (RFP format)", {
   df <- make_example(200, 50)
    g <- 2 / 50

    svm.fitalo <- alocv::alo_svm(df$X, df$y, scale = F, kernel='radial', gamma=g, cost=1,
                                 type='C-classification', tolerance=1e-6, use_rfp=TRUE)

    expect_equal(svm.fitalo$alo_loss, 0.346218829410011)
})


test_that("ALO SVM Correct for with S3 method", {
    df <- make_example(200, 50)
    g <- 2 / 50

    svm_fit <- e1071::svm(df$X, df$y, scale=F, kernel='radial', gamma=g, cost=1,
                          type='C-classification', tolerance=1e-6)

    svm_fit_alo <- alocv::alocv(svm_fit, df$X, df$y)

    expect_equal(svm_fit_alo$alo_loss, 0.346218829410011)
})


test_that("ALO SVM Correct for with S3 method and pivoting", {
    df <- make_example(200, 50)
    g <- 2 / 50

    svm_fit <- e1071::svm(df$X, df$y, scale=F, kernel='radial', gamma=g, cost=1,
                          type='C-classification', tolerance=1e-6)

    svm_fit_alo <- alocv::alocv(svm_fit, df$X, df$y, use_rfp=FALSE, use_pivot=TRUE)

    expect_equal(svm_fit_alo$alo_loss, 0.346218829410011)
})


test_that("Kernel Correct for Polynomial", {
    df <- make_example(20, 10)

    K <- alocv:::compute_svm_kernel(df$X, 1, 0.5, 3, 1.5)
    expected <- as.matrix(tril((0.5 * tcrossprod(df$X) + 1.5)^3))

    expect_equal(matrix(K), matrix(expected))
})

test_that("Kernel Correct for RBF", {
    df <- make_example(20, 10)

    K <- alocv:::compute_svm_kernel(df$X, 0, 0.5, 0, 0)
    expected <- as.matrix(tril(exp(- 0.5 * as.matrix(dist(df$X, diag = T)) ^ 2)))

    expect_equal(matrix(K), matrix(expected))
})

test_that("Kernel Compatible With SVM", {
    df <- make_example(20, 10)

    K <- alocv:::compute_svm_kernel(df$X, 0, 0.5, 0, 0)
    fit_svm <- e1071::svm(df$X, df$y, gamma=0.5, scale=FALSE)

    alpha <- numeric(nrow(df$X))
    alpha[fit_svm$index] <- fit_svm$coefs

    K <- K + t(as.matrix(tril(K, -1)))

    decision_computed <- as.vector(K %*% alpha - fit_svm$rho)
    expect_equal(decision_computed, as.numeric(fit_svm$decision.values))
})
