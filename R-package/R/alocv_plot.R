# Code to plot ALO results

#' Plots information for the given ALO evaluation.
#'
#' @export
plot.alo <- function(x, ...) {
    plot(log(x$lambda), x$alo,
         xlab='lambda',
         ylab='ALO risk')
}
