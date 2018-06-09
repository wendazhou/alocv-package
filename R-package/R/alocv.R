# File with main definitions.

#' alocv: A package for computing unbiased risk estimates for high-dimensional models.
#'
#' @docType package
#' @name alocv
#' @useDynLib alocv
#' @importFrom Rcpp sourceCpp
NULL


#' Computes an approximate-leave-one out risk for the given fit.
#'
#' @param fit An object containing the fitted estimator.
#'
#' @export
alocv <- function(fit, ...) {
    UseMethod('alocv', fit)
}
