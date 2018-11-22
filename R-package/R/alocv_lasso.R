#' Computes approximate leave-one-out errors for each regularization
#' value in the fitted object.
#'
#' @param fit: the fitted glmnet object
#' @param x: the predictor matrix
#' @param y: the response vector
#' @param standardize: indicates whether glmnet was called with standardize. If not specified,
#' this method will try to guess by examining the fitted object.
#' @param intercept: indicates whether glmnet was called with an intercept. If not specified,
#' this method will try to guess by examining the fitted object.
#'
#' @export
alocv.glmnet <- function(fit, x, y, alpha=NULL, standardize=NULL, intercept=NULL, ...) {
    if(is.null(alpha)) {
        warning("Assuming alpha=1 as it was not specified. ",
                "For correctness please ensure value is correctly specified.")
        alpha <- 1
    }

    if(is.null(intercept)) {
        intercept <- (!is.null(fit$a0) && any(fit$a0 != 0))
    }

    if(is.null(standardize)) {
        standardize <- check_standardize(fit, x, y, alpha)
    }

    if(is.null(standardize)) {
        # if still NULL, we couldn't figure out from the fitted object.
        warning("Could not determine whether glmnet object was fitted with standardize. ",
                "Please specify value directly.")
        standardize <- TRUE
    }

    family <- get_glmnet_family(fit)

    alo_glmnet_internal(x, fit$beta, y, fit$lambda, family, alpha, fit$a0)
}

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

    nobs = nrow(x)
    nvars = ncol(x)

    fitted <- glmnet::glmnet(x, y, family, weights, offset, alpha, nlambda,
                             lambda.min.ratio, lambda, standardize, intercept,
                             thresh, dfmax, ...,
                             type.multinomial=type.multinomial)

    alo <- alo_glmnet_internal(x, fitted$beta, y, fitted$lambda, family,
                               alpha, fitted$a0, standardize, intercept)

    class(fitted) <- c("alo", class(fitted))
    fitted$alo <- alo$alo
    fitted$alo_mse <- alo$alo_mse
    fitted$alo_mae <- alo$alo_mae
    fitted$leverage <- alo$leverage

    fitted
}


alo_glmnet_internal <- function(x, beta, y, lambda, family, alpha, a0, standardize, intercept) {
    if(standardize) {
        rescaled <- glmnet_rescale(x, a0, as.matrix(beta), intercept, family)
        x <- rescaled$Xs
        a0 <- rescaled$a0s
        beta <- rescaled$betas
    } else {
        beta <- as.matrix(beta)
    }

    if(family == "gaussian") {
        if(alpha == 1 && !intercept) {
            alo_lasso_rcpp(x, beta, y, has_intercept=intercept)
        } else {
            alo_enet_rcpp(x, beta, y,
                          lambda * lambda_scale(y),
                          family=0, alpha=alpha,
                          has_intercept=intercept, a0=a0)
        }
    } else if(family == "poisson") {
        alo_enet_rcpp(x, beta, y, length(y) * lambda,
                      family=1, alpha=alpha,
                      has_intercept=intercept, a0=a0)
    } else if (family == "binomial") {
        alo_enet_rcpp(x, beta, y, length(y) * lambda,
                      family=2, alpha=alpha,
                      has_intercept=intercept, a0=a0)
    } else {
        stop("Unsupported family")
    }
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

#' Given a glm fit and the data, determines whether
#' standardization was applied to the data before
#' the fit.
#'
#' This check is done by evaluating the dual optimal
#' value and checking whether it corresponds to the
#' stated penalization value.
#'
check_standardize <- function(fit, x, y, alpha) {
    UseMethod("check_standardize")
}

#' Default standardize check, simply returns
#' NULL to indicate that it is not implemented.
check_standardize.default <- function(fit, x, y, alpha) {
    NULL
}

check_standardize.elnet <- function(fit, x, y, alpha) {
    if(!is.null(fit$a0)) {
        # we have an intercept
        y <- scale(y, center=fit$a0[1], scale=FALSE)
    }

    # compute dual optimal under assumption of standardization
    resid0 <- y - x %*% fit$beta[,1]
    lambda <- max(abs(t(x) %*% resid0))

    !isTRUE(all.equal(lambda, fit$lambda[1] * alpha * fit$nobs))
}


#' Simple S3 method to obtain the name of the family
#' used in the fitting method.
get_glmnet_family <- function(fit) {
    UseMethod("get_glmnet_family")
}

get_glmnet_family.default <- function(fit) {
    stop("Unsupported family.")
}

get_glmnet_family.elnet <- function(fit) {
    "gaussian"
}

get_glmnet_family.lognet <- function(fit) {
    "binomial"
}

get_glmnet_family.fishnet <- function(fit) {
    "poisson"
}
