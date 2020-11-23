context("ALO glmnet")
library(alocv)
library(glmnet)

make_example <- function(n, p, eps=0.5, seed=42) {
    df <- withr::with_seed(seed,
        list(x = matrix(rnorm(n * p), nrow=n, ncol=p),
             beta = matrix(rnorm(p) * rbinom(p, 1, eps), ncol=1),
             err = rnorm(n, sd=0.1)))

    df$y <- with(df, x %*% beta + err)
    df
}

test_that("alocv correct for pure lasso with intercept", {
    df <- make_example(20, 10)
    fitted <- alo_glmnet(df$x, df$y, standardize=F, intercept=T, nlambda = 10, lasso_approximate_intercept = F)

    expected_alo <- c(8.83262606, 3.96686767, 0.69026271, 0.13926358, 0.04274022, 0.06060158)

    expect_equal(fitted$alo, expected_alo)
})


test_that("alocv reasonable for pure lasso with intercept approximation", {
    df <- make_example(20, 10)
    fitted <- alo_glmnet(df$x, df$y, standardize=F, intercept=T, nlambda = 10, lasso_approximate_intercept = T)

    expected_alo <- c(8.83262606, 3.96686767, 0.69026271, 0.13926358, 0.04274022, 0.06060158)

    expect_true(all(fitted$alo <= expected_alo))
})

test_that("alocv correct for enet with intercept", {
    data <- make_example(20, 10)
    fitted <- alo_glmnet(data$x, data$y, alpha=0.5, standardize=F, intercept=T)

    expected_alo <- c(
        8.83262606, 8.92694510, 8.37881234, 10.10658713, 9.31379248, 8.57796491,
        8.44628481, 7.55151843, 6.73326064,  5.98893078, 5.31535946, 5.23669348,
        4.57847790, 3.99225617, 3.47246398,  3.01345630, 2.60969679, 2.25583870,
        1.94678759, 1.67774731, 1.44425074,  1.24217729, 1.06775917, 0.91757847,
        0.78855716, 0.67794153, 0.58328295,  0.50241607, 0.43343575, 0.37467354,
        0.32467448, 0.28217465, 0.24608003,  0.21544664, 0.18946241, 0.16743057,
        0.14875471, 0.20021629, 0.16868507,  0.14208721, 0.11972268, 0.10095433,
        0.08523372, 0.07209044, 0.06112220,  0.05198586, 0.04442748, 0.03811811,
        0.03289140, 0.04096408, 0.03560743,  0.03113393)

    expect_equal(fitted$alo, expected_alo)
})

test_that("alocv correct for enet with scaling and intercept", {
    data <- make_example(20, 10)
    fitted <- alo_glmnet(data$x, data$y, alpha=0.5, standardize=T, intercept=T)

    expected_alo <- c(
        8.83262606, 8.93859211, 8.32864850, 7.75758724, 7.22638910, 6.73542747, 6.28450537,
        7.64471781, 7.50387431, 6.67859392, 5.93118178, 5.25737128, 4.65262670, 4.57334243,
        3.98494943, 3.46406614, 3.00480751, 2.60132576, 2.24805021, 1.93973125, 1.67147005,
        1.43873563, 1.23737138, 1.06359257, 0.91397692, 0.78544961, 0.67526432, 0.58098157,
        0.50044544, 0.43175956, 0.37326310, 0.32350742, 0.28123360, 0.24535142, 0.21491967,
        0.18912818, 0.16728144, 0.23217077, 0.19475287, 0.16337712, 0.13711118, 0.11517987,
        0.09691583, 0.08174650, 0.06918226, 0.05880566, 0.05026180, 0.04329208, 0.03755120,
        0.03287328, 0.02907830, 0.03194108, 0.02837800)

    expect_equal(fitted$alo, expected_alo)
})

test_that("alocv S3 method correct for enet", {
    df <- make_example(20, 10)
    fit_glmnet <- glmnet::glmnet(df$x, df$y, alpha=0.5)
    fitted_alo <- alocv(fit_glmnet, df$x, df$y, alpha=0.5)

    expected_alo <- c(
        8.83262606, 8.93859211, 8.32864850, 7.75758724, 7.22638910, 6.73542747, 6.28450537,
        7.64471781, 7.50387431, 6.67859392, 5.93118178, 5.25737128, 4.65262670, 4.57334243,
        3.98494943, 3.46406614, 3.00480751, 2.60132576, 2.24805021, 1.93973125, 1.67147005,
        1.43873563, 1.23737138, 1.06359257, 0.91397692, 0.78544961, 0.67526432, 0.58098157,
        0.50044544, 0.43175956, 0.37326310, 0.32350742, 0.28123360, 0.24535142, 0.21491967,
        0.18912818, 0.16728144, 0.23217077, 0.19475287, 0.16337712, 0.13711118, 0.11517987,
        0.09691583, 0.08174650, 0.06918226, 0.05880566, 0.05026180, 0.04329208, 0.03755120,
        0.03287328, 0.02907830, 0.03194108, 0.02837800)

    expect_equal(fitted_alo$alo, expected_alo)
})

make_example_poisson <- function(n, p, eps=0.5, seed=42) {
    data <- withr::with_seed(seed, {
        x <- matrix(rnorm(n * p), nrow=n, ncol=p)
        beta <- matrix(rnorm(p) * rbinom(p, 1, eps), ncol=1)
        y <- rpois(n, exp(x %*% beta))
        list(x=x, beta=beta, y=y)
    })

    data
}

test_that("alocv correct for poisson enet", {
    data <- make_example_poisson(200, 10)
    fitted <- alo_glmnet(data$x, data$y, alpha=0.5, family="poisson",
                         standardize=F, intercept=F, nlambda=10)

    expected_alo <- c(
        50.4907702, 22.4555737, 5.7638338, 1.8894571, 0.9353201, 0.7927818,
        0.7928884, 0.8012724, 0.7978407, 0.7971896)

    expect_equal(fitted$alo, expected_alo)
})


make_example_logit <- function(n, p, eps=0.5, seed=42) {
    data <- withr::with_seed(seed, {
        x <- matrix(rnorm(n * p), nrow=n, ncol=p)
        beta <- matrix(rnorm(p) * rbinom(p, 1, eps), ncol=1)
        y <- rbinom(n, 1, 1 / (1 + exp(-x %*% beta)))
        list(x=x, beta=beta, y=y)
    })

    data
}

test_that("alocv correct for logistic glmnet no standardize", {
    data <- make_example_logit(50, 10)

    fitted <- alo_glmnet(data$x, data$y, alpha=0.5, family="binomial",
                         standardize=F, intercept=F, nlambda = 10)

    expected_alo <- c(
        0.5000000, 0.4245442, 0.3758924, 0.4097286, 0.4408108, 0.4515035,
        0.4567730, 0.4850738, 0.4861418, 0.4865248)

    expect_equal(fitted$alo_mse, expected_alo, tolerance=1e-6)
})


test_that("alocv correct for logistic glmnet with standardize", {
    data <- make_example_logit(50, 10)

    fitted <- alo_glmnet(data$x, data$y, alpha=0.5, family="binomial",
                         standardize=T, intercept=F, nlambda = 10)

    expected_alo <- c(
        0.5000000, 0.4168007, 0.4052387, 0.4130048, 0.4437558,
        0.4529238, 0.4573545, 0.4853284, 0.4862312, 0.4865597)

    expect_equal(fitted$alo_mse, expected_alo, tolerance=1e-6)
})


elnet_standardize_test <- function(intercept, standardize, alpha=1) {
    df <- make_example(20, 10)
    fit <- glmnet(df$x, df$y, standardize = standardize, intercept = intercept, alpha = alpha)
    check_standardize(fit, df$x, df$y, alpha=alpha)
}

test_that("check standardize correct for elnet (no intercept / standardize)", {
    expect_true(elnet_standardize_test(intercept = F, standardize = T))
})

test_that("check standardize correct for elnet (no intercept / no standardize)", {
    expect_false(elnet_standardize_test(intercept = F, standardize = F))
})

test_that("check standardize correct for elnet (intercept / standardize)", {
    expect_true(elnet_standardize_test(intercept = T, standardize = T))
})

test_that("check standardize correct for elnet (intercept / no standardize)", {
    expect_false(elnet_standardize_test(intercept = T, standardize = F))
})


test_that("check standardize correct for elnet (no intercept / standardize / alpha)", {
    expect_true(elnet_standardize_test(intercept = F, standardize = T, alpha = 0.5))
})

test_that("check standardize correct for elnet (no intercept / no standardize / alpha)", {
    expect_false(elnet_standardize_test(intercept = F, standardize = F, alpha = 0.5))
})

test_that("check standardize correct for elnet (intercept / standardize / alpha)", {
    expect_true(elnet_standardize_test(intercept = T, standardize = T, alpha = 0.5))
})

test_that("check standardize correct for elnet (intercept / no standardize / alpha)", {
    expect_false(elnet_standardize_test(intercept = T, standardize = F, alpha = 0.5))
})

