# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' Computes alo values for the LASSO problem using a up/down-dating
#' algorithm to reuse intermediate values along a given regularization
#' path.
#'
#' Note that due to the special algorithm, it is not strictly correct
#' for models fitted with an intercept, as it does not account for
#' the uncertaining in estimating the intercept.
#'
#' @param A The predictor matrix (\code{n x p}).
#' @param B The matrix of fitted values for each regularization (\code{p x m})
#' @param y The vector of observed responses (length \code{n}).
#' @param a0 An optional vector of fitted intercepts.
#'
#' @return A list with the computed alo values and leverage.
#'
alo_lasso_rcpp <- function(A, B, y, a0 = NULL) {
    .Call('_alocv_alo_lasso_rcpp', PACKAGE = 'alocv', A, B, y, a0)
}

#' Computes alo values for the elastic-net problem (and its GLM forms).
#' Note that this function is significantly slower than \code{\link{alo_lasso_rcpp}}
#' for large problems, as it must solve each problem separately.
#'
#' @param A The predictor matrix
#' @param B The matrix of estimated coefficients for each tuning value.
#' @param y The vector of observed responses.
#' @param lambda The vector of tuning values.
#' @param family An enumeration indicating the GLM family.
#' @param alpha The elastic-net parameter (balance between l1 and l2 penalty).
#' @param a0 If not NULL, the estimated intercept values.
#' @param tolerance The numerical tolerance for detecting active sets.
#' @param use_rfp If true, uses a more compact storage format for symmetric matrices.
#'
alo_enet_rcpp <- function(A, B, y, lambda, family = 0L, alpha = 1.0, a0 = NULL, tolerance = 1e-5, use_rfp = TRUE) {
    .Call('_alocv_alo_enet_rcpp', PACKAGE = 'alocv', A, B, y, lambda, family, alpha, a0, tolerance, use_rfp)
}

alo_svm_kernel_rcpp <- function(K, y, alpha, rho, lambda, tolerance = 1e-5, use_rfp = FALSE, use_pivot = FALSE) {
    .Call('_alocv_alo_svm_kernel_rcpp', PACKAGE = 'alocv', K, y, alpha, rho, lambda, tolerance, use_rfp, use_pivot)
}

compute_svm_kernel <- function(X, kernel_type, gamma, degree, coef0, use_rfp = FALSE) {
    .Call('_alocv_compute_svm_kernel', PACKAGE = 'alocv', X, kernel_type, gamma, degree, coef0, use_rfp)
}

alo_svc_rcpp <- function(X, y, alpha, rho, lambda, kernel_type, gamma, degree, coef0, tolerance = 1e-5, use_rfp = FALSE, use_pivot = FALSE) {
    .Call('_alocv_alo_svc_rcpp', PACKAGE = 'alocv', X, y, alpha, rho, lambda, kernel_type, gamma, degree, coef0, tolerance, use_rfp, use_pivot)
}

alo_svr_rcpp <- function(X, y, alpha, rho, lambda, epsilon, kernel_type, gamma, degree, coef0, tolerance = 1e-5, use_rfp = FALSE, use_pivot = FALSE) {
    .Call('_alocv_alo_svr_rcpp', PACKAGE = 'alocv', X, y, alpha, rho, lambda, epsilon, kernel_type, gamma, degree, coef0, tolerance, use_rfp, use_pivot)
}

