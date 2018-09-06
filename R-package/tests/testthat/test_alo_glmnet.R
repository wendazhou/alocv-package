context("ALO glmnet")
library(alocv)
library(glmnet)

make_example <- function(n, p, eps=0.5, seed=42) {
    data <- withr::with_seed(seed,
        list(x = matrix(rnorm(n * p), nrow=n, ncol=p),
             beta = matrix(rnorm(p) * rbinom(p, 1, eps), ncol=1),
             err = rnorm(n, sd=0.1)))

    data$y <- with(data, x %*% beta + err)
    data
}

test_that("alocv correct for pure lasso", {
    data <- make_example(20, 10)

    fitted <- alo.glmnet(data$x, data$y, standardize=F, intercept=F)

    expected_alo <- c(
        7.99425746, 7.96533423, 7.05010423, 6.29263845, 8.95506254, 7.85682146, 6.93893283,
        6.88841325, 5.80323213, 4.90017479, 4.14858414, 3.52290875, 3.00192071, 2.88942461,
        2.41472635, 2.01948633, 1.69060507, 1.41688237, 1.18911446, 0.99935316, 0.84129565,
        0.70960499, 0.59984618, 0.50833351, 0.43200365, 0.36831032, 0.31513675, 0.27072300,
        0.23360563, 0.20256755, 0.17659637, 0.15484985, 0.13662711, 0.12134486, 0.10851747,
        0.09776550, 0.08869874, 0.12607060, 0.10514547)

    expect_equal(fitted$alo, expected_alo)
})
