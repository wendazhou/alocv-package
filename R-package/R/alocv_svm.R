
#' Fits and computes the approximate cross-validation for SVMs.
#'
#'
#'@export
alo.svm <- function(x, y, scale = TRUE, type = NULL,
                    kernel = "radial", degree = 3, gamma = if(is.vector(x)) 1 else 1 / ncol(x),
                    coef0 = 0, cost = 1, nu = 0.5, use_rfp = TRUE) {
    svm.fit <- e1071::svm(x, y, scale, type, kernel, degree=degree, gamma=gamma, coef0=coef0, cost=cost)
    kernel_type <- charmatch(kernel, c("radial", "polynomial", "linear"))

    if(is.na(kernel_type) || kernel_type == 0) {
        stop("Could not find given kernel type")
    }

    K <- alo_svm_kernel(x, kernel_type - 1, gamma, degree, coef0, use_rfp=use_rfp)
    alo_svm_rcpp(K, y, svm.fit$fitted, svm.fit$rho, 1 / svm.fit$cost, tolerance = 1e-5, use_rfp=use_rfp)
}
