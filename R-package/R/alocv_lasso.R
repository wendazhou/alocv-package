#' Computes approximate leave-one-out errors for each regularization
#' value in the fitted object.
#'
#' @param fit the fitted glmnet object
#' @param x the predictor matrix
#' @param y the response vector
#' @param alpha the elastic-net parameter used to fit the model.
#' @param standardize indicates whether \code{\link[glmnet]{glmnet}} was called with standardize.
#' If not specified, this method will try to guess by examining the fitted object.
#' @param intercept indicates whether \code{\link[glmnet]{glmnet}} was called with an intercept.
#' If not specified, this method will try to guess by examining the fitted object.
#' @param ... Further arguments passed to or from other methods.
#' @param lasso_approximate_intercept If true, uses a slightly less accurate but
#' faster method for the pure LASSO case (which ignores the variability of the
#' intercept).
#'
#' @seealso \code{\link[glmnet]{glmnet}}
#'
#' @export
alocv.glmnet <- function(fit, x, y, alpha=NULL, standardize=NULL, intercept=NULL,
                         ..., lasso_approximate_intercept=TRUE) {
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

    alo <- alo_glmnet_internal(x, fit$beta, y, fit$lambda, family, alpha,
                               fit$a0, standardize, intercept, lasso_approximate_intercept)

    class(fit) <- c("alo_glmnet", "alo", class(fit))
    fit$alo <- alo$alo
    fit$alo_mse <- alo$alo_mse
    fit$alo_mae <- alo$alo_mae
    fit$leverage <- alo$leverage

    fit
}

#' Fits and computes the approximate leave-one-out cross validation for glmnet.
#'
#' @param x The data matrix.
#' @param y The response vector.
#' @param family The family of the glm to fit.
#' @param weights Observation weights.
#' @param offset A vector of length nobs included in the linear predictor.
#' @param alpha Elastic net parameter.
#' @param nlambda The number of \code{lambda} values.
#' @param lambda.min.ratio Smallest value for \code{lambda} as a fraction of the largest.
#' @param lambda A user supplied lambda sequence.
#' @param standardize Whether to apply \code{x} variable standardization.
#' @param intercept Whether intercepts should be fitted.
#' @param thresh Convergence threshold for coordinate descent.
#' @param dfmax Limit maximum number of variables in the model.
#' @param ... Other parameters passed on to \code{\link[glmnet]{glmnet}}
#' @param lasso_approximate_intercept If true, uses a slightly less accurate but
#' faster method for the pure LASSO case (which ignores the variability of the
#' intercept).
#'
#' @return An object with S3 class \code{"alo"} in addition to the
#' classes corresponding to the fitted object returned by \code{\link[glmnet]{glmnet}}.
#' Additional fields are introduced to the object to represent the computed ALO
#' values.
#' \item{alo}{A numerical vector representing the ALO deviance loss for each tuning value.}
#' \item{alo_mse}{A numerical vector representing the ALO MSE loss for each tuning value.}
#' \item{alo_mae}{A numerical vector representing the ALO MAE loss for each tuning value.}
#' \item{leverage}{A numerical matrix representing the computed leverage of each data point for each tuning value.}
#'
#' @seealso \code{\link[glmnet]{glmnet}}
#'
#' @export
alo_glmnet <- function(x, y, family=c("gaussian", "binomial", "poisson"),
                       weights, offset=NULL, alpha=1, nlambda=100,
                       lambda.min.ratio = ifelse(nobs<nvars,0.01, 0.0001),
                       lambda=NULL, standardize=TRUE, intercept=TRUE, thresh=1e-7,
                       dfmax = nvars + 1, ..., lasso_approximate_intercept=TRUE) {
    family = match.arg(family)
    alpha = as.double(alpha)

    nobs = nrow(x)
    nvars = ncol(x)

    if (!requireNamespace("glmnet", quietly = TRUE)) {
        stop("Package \"glmnet\" is required for this function to work. Please install it.", call. = FALSE)
    }
    fitted <- glmnet::glmnet(x, y, family, weights, offset, alpha, nlambda,
                             lambda.min.ratio, lambda, standardize, intercept,
                             thresh, dfmax, ...)

    alo <- alo_glmnet_internal(x, fitted$beta, y, fitted$lambda, family,
                               alpha, fitted$a0, standardize, intercept, lasso_approximate_intercept)

    class(fitted) <- c("alo_glmnet", "alo", class(fitted))
    fitted$alo <- alo$alo
    fitted$alo_mse <- alo$alo_mse
    fitted$alo_mae <- alo$alo_mae
    fitted$leverage <- alo$leverage

    fitted
}


#' Internal code to compute ALO for Lasso / elastic net problems.
#'
#' @param x The data matrix
#' @param beta A matrix of fitted parameter values for each tuning value.
#' @param y The response vector.
#' @param lambda A vector of penalization values.
#' @param family The family of the glm that was fit.
#' @param alpha The elastic-net parameter.
#' @param a0 The fitted intercept values for each penalization value.
#' @param standardize A logical value indicating whether \code{x} should be standardized.
#' @param intercept A logical value indicating whether an intercept was fit to the model.
#' @param lasso_approximate_intercept If true, uses a faster algorithm for the pure lasso
#' case even when the intercept is given. This ignores the contribution of the uncertainty
#' of the intercept to the final estimation error.
#'
#'
alo_glmnet_internal <- function(x, beta, y, lambda, family, alpha, a0,
                                standardize, intercept, lasso_approximate_intercept=FALSE) {
    if(standardize) {
        rescaled <- glmnet_rescale(x, a0, as.matrix(beta), intercept, family)
        x <- rescaled$Xs
        a0 <- rescaled$a0s
        beta <- rescaled$betas
    } else {
        beta <- as.matrix(beta)
    }

    if(family == "gaussian") {
        if(alpha == 1 && (!intercept || lasso_approximate_intercept)) {
            # Special fast-path algorithm for LASSO.
            alo_lasso_rcpp(x, beta, y, a0=if(intercept) a0 else NULL)
        } else {
            alo_enet_rcpp(x, beta, y,
                          lambda * lambda_scale(y),
                          family=0, alpha=alpha,
                          a0=if(intercept) a0 else NULL)
        }
    } else if(family == "poisson") {
        alo_enet_rcpp(x, beta, y, length(y) * lambda,
                      family=1, alpha=alpha,
                      a0=if(intercept) a0 else NULL)
    } else if (family == "binomial") {
        alo_enet_rcpp(x, beta, y, length(y) * lambda,
                      family=2, alpha=alpha,
                      a0=if(intercept) a0 else NULL)
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
#' @param fit The fitted object
#' @param x The data matrix
#' @param y The response vector
#' @param alpha The elastic net parameter.
check_standardize <- function(fit, x, y, alpha) {
    UseMethod("check_standardize")
}

# Default standardize check, simply returns
# NULL to indicate that it is not implemented.
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


# Simple S3 method to obtain the name of the family
# used in the fitting method.
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
