#' Computes the approximate leave one out estimator for elastic-net
#' estimators.
#'
#' This function computes the approximate leave-one-out risk and leverage
#' along a path produced by a call to `glmnet`.
#'
#' @export
alocv.elnet <- function(fit, ...) {
    args <- list(...)

    if ('x' %in% names(args)) {
        A <- args[['x']]
    } else {
        A <- eval(fit$call[['x']])
    }

    if ('y' %in% names(args)) {
        y <- args[['y']]
    } else {
        y <- eval(fit$call[['y']])
    }

    B <- as.matrix(fit$beta)

    result <- alo_lasso_rcpp(A, B, y)
    result[['original_fit']] <- fit
    result[['lambda']] <- fit[['lambda']]
    class(result) <- c('alo', 'alo.elnet')
    result
}
