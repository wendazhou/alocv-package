# File with main definitions.

#' alocv: A package for computing unbiased risk estimates for high-dimensional models.
#'
#' @docType package
#' @name alocv
#' @useDynLib alocv
#' @importFrom Rcpp sourceCpp
#' @importFrom graphics plot
NULL


#' Computes the approximate cross-validation value for a fitted
#' classifier or regressor.
#'
#' @param fit The fitted object.
#' @param x The predictor or feature matrix.
#' @param y The response vector.
#' @param ... Arguments to be passed to methods.
#'
#' @seealso \code{\link{alocv.glmnet}}
#' @seealso \code{\link{alocv.svm}}
#'
#' @export
alocv <- function(fit, x, y, ...) {
    UseMethod("alocv", fit)
}
