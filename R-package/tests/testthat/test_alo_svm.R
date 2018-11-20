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

    svm.fitalo <- alocv::alo.svm(df$X, df$y, scale = F, kernel='radial', gamma=g, cost=1,
                                 type='C-classification', use_rfp=FALSE)

    expect_equal(svm.fitalo$loss, 0.346218829410011)
})


test_that("ALO SVM Correct for Gaussian RBF (RFP format)", {
    df <- make_example(200, 50)
    g <- 2 / 50

    svm.fitalo <- alocv::alo.svm(df$X, df$y, scale = F, kernel='radial', gamma=g, cost=1,
                                 type='C-classification', use_rfp=TRUE)

    expect_equal(svm.fitalo$loss, 0.346218829410011)
})
