
#' Fits and computes the approximate cross-validation for SVMs.
#'
#' @param x The data matrix
#' @param y The response vector
#' @param scale A logical vector indicating the variables to be scaled.
#' @param type The type of svm to fit. Only 'C-classification' is supported at the moment.
#' @param kernel The kernel used in traaining and predicting.
#' @param degree Parameter for polynomial kernels.
#' @param gamma Scale parameter needed for all kernels. Default 1 / (data dimension).
#' @param coef0 Offset parameter needed for kernels.
#' @param cost Cost of constrain violations.
#' @param nu Hyperparameter for some SVM types.
#' @param tolerance Tolerance for SVM solver, and for support vector detection in ALO procedure.
#' @param use_rfp If true, uses rectangular full packed matrices in ALO algorithm. Reduces memory requirement.
#' @param use_pivot If true, uses pivoting in Cholesky factorization of kernel matrix.
#' @param ... Additional parameters to be passed to the underlying SVM fit function.
#'
#' @seealso \code{\link[e1071]{svm}}
#' @importFrom e1071 svm
#' @export
alo_svm <- function(x, y, scale = TRUE, type = NULL,
                    kernel = "radial", degree = 3, gamma = if(is.vector(x)) 1 else 1 / ncol(x),
                    coef0 = 0, cost = 1, nu = 0.5, tolerance=1e-4, use_rfp = TRUE, use_pivot = FALSE, ...) {

    if (!requireNamespace("e1071", quietly = TRUE)) {
        stop("Package \"e1071\" is required for this function to work. Please install it.", call. = FALSE)
    }

    fit <- svm(x, y, scale=scale, type=type, kernel=kernel, degree=degree, gamma=gamma,
               coef0=coef0, cost=cost, tolerance=tolerance, ...)

    alocv.svm(fit, x, y, tolerance=tolerance, use_rfp=use_rfp, use_pivot=use_pivot)
}

#' Approximate Leave-one-out cross validation for SVM objects
#'
#' @param fit The fitted svm object.
#' @param x The data matrix.
#' @param y The response vector.
#' @param tolerance The tolerance to use in detecting support vectors.
#' @param use_rfp Whether to use rectangular full packed format to represent the kernel. Reduces memory usage.
#' @param use_pivot Whether to use pivoted Cholesky decomposition. Set this to true when the kernel
#' is not strictly positive-definite.
#' @param ... Further arguments passed to or from other methods.
#'
#' @export
alocv.svm <- function(fit, x, y, tolerance=1e-5, use_rfp=TRUE, use_pivot=FALSE, ...) {
    alpha <- numeric(nrow(x))
    alpha[fit$index] <- fit$coefs

    if(!is.null(fit$x.scale)) {
        x <- base::scale(x, center=fit$x.scale$`scaled:center`,
                         scale=fit$x.scale$`scaled:scale`)
    }

    # convert y to factors first, if not already
    # note e1071::svm will convert it internally for classification,
    # but e1071::tune.svm won't!
    if (!is.factor(y)) {
        y = as.factor(y)
    }

    # convert factors back to numeric -1, 1
    # as in LIBSVM the first element must be 1
    y = 3 - 2 * as.numeric(y)
    y = y * y[1]

    kernel <- c("linear", "polynomial", "radial", "sigmoid")[fit$kernel + 1]

    if(use_pivot) {
        # Pivot incompatible with RFP.
        use_rfp <- FALSE
    }

    if(kernel == "linear" && nrow(x) > ncol(x)) {
        if(!use_pivot) {
            warning("Forcing use_pivot to true as linear kernel with n > p is singular.")
        }

        use_rfp <- FALSE
        use_pivot <- TRUE
    }

    alo_info <- alo_svm_internal(x, y, alpha, fit$rho, 1 / fit$cost, kernel,
                                fit$gamma, fit$degree, fit$coef0,
                                tolerance, use_rfp, use_pivot)

    fit$alo_loss <- alo_info$loss
    fit$alo_predicted <- alo_info$predicted

    fit
}

#' Approximate Leave-one-out cross validation for SVM objects
#'
#' @param fit The fitted svm object.
#' @param x The data matrix.
#' @param y The response vector.
#' @param tolerance The tolerance to use in detecting support vectors.
#' @param use_rfp Whether to use rectangular full packed format to represent the kernel. Reduces memory usage.
#' @param use_pivot Whether to use pivoted Cholesky decomposition. Set this to true when the kernel
#' is not strictly positive-definite.
#' @param ... Further arguments passed to or from other methods.
#'
#' @export
alo_svm_internal <- function(x, y, alpha, rho, lambda, kernel = "radial",
                             gamma = if(is.vector(x)) 1 else 1 / ncol(x),
                             degree = 3, coef0 = 0, tolerance=1e-4,
                             use_rfp=TRUE, use_pivot=FALSE) {

    kernel_type <- charmatch(kernel, c("radial", "polynomial", "linear"))
    if(is.na(kernel_type) || kernel_type == 0) {
        stop("Could not find given kernel type")
    }

    if(use_rfp && use_pivot) {
        warning("Both use_rfp and use_pivot are specified")
    }

    alo_info <- alo_svc_rcpp(x, y, alpha, rho, lambda, kernel_type - 1, gamma, degree, coef0,
                             tolerance, use_rfp, use_pivot)

    alo_info
}
