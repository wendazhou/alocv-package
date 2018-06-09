# Code to plot ALO results

#' Plots information for the given ALO evaluation.
#'
#' @export
plot.alo <- function(alo) {
    plot(log(alo$lambda), alo$alo,
         xlab='lambda',
         ylab='ALO risk')
}
