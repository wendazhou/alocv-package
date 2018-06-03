#' @useDynLib alocv
#' @importFrom Rcpp sourceCpp
NULL

#' @export
alocv.elnet <- function(fit) {
    A <- eval(fit$call[['x']])
    y <- eval(fit$call[['y']])
    B <- as.matrix(fit$beta)

    alo_lasso_rcpp(A, B, y)
}
