#' Computes the approximate leave one out estimator for elastic-net
#' estimators.
#'
#' This function computes the approximate leave-one-out risk and leverage
#' along a path produced by a call to `glmnet`.
#'
#' @export
alocv.elnet <- function(fit) {
    A <- eval(fit$call[['x']])
    y <- eval(fit$call[['y']])
    B <- as.matrix(fit$beta)

    alo_lasso_rcpp(A, B, y)
}
