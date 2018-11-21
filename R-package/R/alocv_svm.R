
#' Fits and computes the approximate cross-validation for SVMs.
#'
#'
#'@export
alo_svm <- function(x, y, scale = TRUE, type = NULL,
                    kernel = "radial", degree = 3, gamma = if(is.vector(x)) 1 else 1 / ncol(x),
                    coef0 = 0, cost = 1, nu = 0.5, tolerance=1e-4, use_rfp = TRUE) {
    fit <- e1071::svm(x, y, scale=scale, type=type, kernel=kernel, degree=degree, gamma=gamma,
                      coef0=coef0, cost=cost, tolerance=tolerance)

    alocv.svm(fit, x, y, tolerance=tolerance, use_rfp=use_rfp)
}

#' Approximate Leave-one-out cross validation for SVM objects
#'
#' @param tolerance: The tolerance to use in detecting support vectors
#' @param use_rfp: Whether to use rectangular full packed format to represent the kernel
#'
#' @export
alocv.svm <- function(fit, x, y, tolerance=1e-5, use_rfp=TRUE) {
    alpha <- numeric(nrow(x))
    alpha[fit$index] <- fit$coefs

    if(!is.null(fit$x.scale)) {
        x <- base::scale(x, center=svm.fit$x.scale$`scaled:center`,
                         scale=svm.fit$x.scale$`scaled:scale`)
    }

    kernel <- c("linear", "polynomial", "radial", "sigmoid")[fit$kernel + 1]

    alo_info <- alo_svm_internal(x, y, alpha, fit$rho, 1 / fit$cost, kernel,
                                fit$gamma, fit$degree, fit$coef0,
                                tolerance, use_rfp)

    fit$alo_loss <- alo_info$loss
    fit$alo_predicted <- alo_info$predicted

    fit
}

alo_svm_internal <- function(x, y, alpha, rho, lambda, kernel = "radial",
                             gamma = if(is.vector(x)) 1 else 1 / ncol(x),
                             degree = 3, coef0 = 0, tolerance=1e-4, use_rfp=TRUE) {

    kernel_type <- charmatch(kernel, c("radial", "polynomial", "linear"))
    if(is.na(kernel_type) || kernel_type == 0) {
        stop("Could not find given kernel type")
    }

    alo_info <- alo_svm_rcpp(x, y, alpha, rho, lambda, kernel_type - 1, gamma, degree, coef0,
                             tolerance, use_rfp)

    alo_info
}
