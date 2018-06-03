#' @useDynLib alocv
#' @importFrom Rcpp sourceCpp
NULL

compute_h_lasso_path <- function(x, y, beta_hats) {
}

compute_h_lasso <- function(x, y, beta_hat, tol=1e-6) {
    E <- abs(beta_hat) > tol
    W <- x[,E]

    S <- t(W) %*% W
    K <- chol(S)

    h <- colSums(backsolve(K, t(W), transpose = TRUE) ^ 2)
    h
}

update_h_lasso <- function(x, y, K, E_new, E_old) {
}
