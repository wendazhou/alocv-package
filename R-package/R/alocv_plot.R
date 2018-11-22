# Code to plot ALO results

#' Plots information for the given ALO evaluation.
#'
#' @param x The alo object to plot.
#' @param ... Additional arguments to be passed to the underlying plotting function.
#'
#' @export
plot.alo_glmnet <- function(x, ...) {
    plot(x$lambda, x$alo,
         xlab='lambda',
         ylab='ALO deviance risk',
         log="x", ...)
}
