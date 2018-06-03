# File with main definitions.

#' @useDynLib alocv
#' @importFrom Rcpp sourceCpp


#' @export
alocv <- function(fit) {
    UseMethod('alocv', fit)
}
