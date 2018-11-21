array_to_c <- function(x, name) {
    paste0('static const double ', name, '[] = {',
            paste(x, collapse=','), '};')
}

constant_to_c <- function(x, name) {
    if(class(x) == "integer") {
        c_class <- "int";
    } else {
        c_class <- "double"
    }
    paste0('const ', c_class, " ", name, ' = ', x, ';');
}

values_to_c <- function(...) {
    values <- list(...)
    name <- names(values)

    lines <- lapply(seq_along(values), function(i) {
        v <- values[[i]]
        n <- name[[i]]

        if (length(v) == 1) {
            constant_to_c(v, n)
        } else {
            array_to_c(v, n)
        }
    })

    paste(lines, collapse='\n')
}
