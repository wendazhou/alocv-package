
#' Fits and computes the approximate cross-validation for SVMs.
#'
#'
#'@export
alo.svm <- function(x, y, scale = TRUE, type = NULL,
                    kernel = "radial", degree = 3, gamma = if(is.vector(x)) 1 else 1 / ncol(x),
                    coef0 = 0, cost = 1, nu = 0.5, use_rfp = TRUE) {
    fit <- e1071::svm(y ~ x, scale=scale, type=type, kernel=kernel, degree=degree, gamma=gamma,
                               coef0=coef0, cost=cost, tolerance=1e-6)
    kernel_type <- charmatch(kernel, c("radial", "polynomial", "linear"))

    if(is.na(kernel_type) || kernel_type == 0) {
        stop("Could not find given kernel type")
    }

    K <- alo_svm_kernel(x, kernel_type - 1, gamma, degree, coef0, use_rfp=use_rfp)

    alpha <- rep(0, nrow(x))
    alpha[svm.fit$index] <- fit$coefs

    alo_info <- alo_svm_rcpp(K, y, alpha, fit$rho, 1 / svm.fit$cost, tolerance = 1e-5, use_rfp=use_rfp)

    fit$alo_loss <- alo_info$loss
    fit$alo_predicted <- alo_info$predicted

    fit
}
