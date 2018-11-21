

#' Fits and computes the approximate leave-one-out cross validation for glmnet.
#'
#' See glmnet for parameter description
#'
#' @export
alo_glmnet <- function(x, y, family=c("gaussian", "binomial", "poisson", "multinomial"),
                       weights, offset=NULL, alpha=1, nlambda=100,
                       lambda.min.ratio = ifelse(nobs<nvars,0.01, 0.0001),
                       lambda=NULL, standardize=TRUE, intercept=TRUE, thresh=1e-7,
                       dfmax = nvars + 1, type.multinomial=c("ungrouped", "grouped"),
                       ...) {
    family = match.arg(family)
    alpha = as.double(alpha)
    nlam = as.integer(nlambda)

    nobs = nrow(x)
    nvars = ncol(x)

    fitted <- glmnet::glmnet(x, y, family, weights, offset, alpha, nlambda,
                             lambda.min.ratio, lambda, standardize, intercept,
                             thresh, dfmax, ...,
                             type.multinomial=type.multinomial)

    if(standardize) {
        rescaled <- glmnet_rescale(x, fitted$a0, as.matrix(fitted$beta), intercept, family)
        x <- rescaled$Xs
        a0 <- rescaled$a0s
        beta <- rescaled$betas
    } else {
        a0 <- fitted$a0
        beta <- as.matrix(fitted$beta)
    }

    if(family == "gaussian") {
        if(alpha == 1 && !intercept) {
            alo <- alo_lasso_rcpp(x, beta, y, has_intercept=intercept)
        } else {
            alo <- alo_enet_rcpp(x, beta, y,
                                 fitted$lambda * lambda_scale(y),
                                 family=0, alpha=alpha,
                                 has_intercept=intercept, a0=a0)
        }
    } else if(family == "poisson") {
        alo <- alo_enet_rcpp(x, beta, y, length(y) * fitted$lambda,
                             family=1, alpha=alpha,
                             has_intercept=intercept, a0=a0)
    } else if (family == "binomial") {
        alo <- alo_enet_rcpp(x, beta, y, length(y) * fitted$lambda,
                             family=2, alpha=alpha,
                             has_intercept=intercept, a0=a0)
    } else {
        stop("Unsupported family")
    }

    class(fitted) <- c("alo", class(fitted))
    fitted$alo <- alo$alo
    fitted$alo_mse <- alo$alo_mse
    fitted$alo_mae <- alo$alo_mae
    fitted$leverage <- alo$leverage

    fitted
}

glmnet_rescale = function(X, a0, beta, intercept, family) {
    n = nrow(X)
    mean_X = colMeans(X)
    sd_X = sqrt(colSums(X^2 / n) - colSums(X / n)^2)
    X = scale(X, center = intercept, scale = sd_X) # no centering if no intercept

    if (family == "multinomial") {
        beta = lapply(beta, "*", sd_X)
        for (i in 1:nrow(a0)) {
            a0[i, ] = as.vector(a0[i, ] + mean_X %*% beta[[i]])
        }
    } else {
        beta = beta * sd_X
        a0 = as.vector(a0 + mean_X %*% beta)
    }

    return(list(Xs = X, a0s = a0, betas = beta))
}

lambda_scale <- function(y) {
    n <- length(y)
    sy <- sqrt(mean((y - mean(y)) ^ 2))

    n / sy
}
