
#' Fits and computes the approximate cross-validation for SVMs.
#'
#'
#'@export
alo.svm <- function(x, y, scale = TRUE, type = NULL,
                    kernel = "radial", degree = 3, gamma = if(is.vector(x)) 1 else 1 / ncol(x),
                    coef0 = 0, cost = 1, nu = 0.5, tolerance=1e-4, use_rfp = TRUE) {
    fit <- e1071::svm(x, y, scale=scale, type=type, kernel=kernel, degree=degree, gamma=gamma,
                      coef0=coef0, cost=cost, tolerance=tolerance)
    kernel_type <- charmatch(kernel, c("radial", "polynomial", "linear"))

    if(!is.null(fit$x.scale)) {
        x <- base::scale(x, center=svm.fit$x.scale$`scaled:center`,
                         scale=svm.fit$x.scale$`scaled:scale`)
    }

    if(is.na(kernel_type) || kernel_type == 0) {
        stop("Could not find given kernel type")
    }

    alpha <- rep(0, nrow(x))
    alpha[fit$index] <- fit$coefs

    alo_info <- alo_svm_rcpp(x, y, alpha, fit$rho, 1 / fit$cost,
                             kernel_type - 1, gamma, degree, coef0,
                             tolerance=tolerance, use_rfp=use_rfp)

    fit$alo_loss <- alo_info$loss
    fit$alo_predicted <- alo_info$predicted

    fit
}
