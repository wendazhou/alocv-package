#! /usr/env Rscript --no-restore --no-save

library(glmnet)

args <- commandArgs(TRUE)

if(!(length(args) %in% c(3, 4))) {
    print("Expected three or 4 arguments arguments")
    quit(status = 1)
}

n <- as.integer(args[1])
p <- as.integer(args[2])
out_name <- args[-1]

if(length(args) == 4) {
    family <- args[3]
} else {
    family <- "gaussian"
}

config <- switch(
    family,
    gaussian = list(link = function(x) { x },
                    rng = function(x) { rnorm(length(x), mean=x, sd=0.1)}),
    binomial = list(link = function(x) { 1 / (1 + exp(x)) },
                    rng = function(x) { rbinom(length(x), 1, prob=x)}),
    poisson = list(link = exp,
                   rng = function(x) { rpois(length(x), 10 * x) }))


X <- matrix(rnorm(n * p, sd = 1 / sqrt(n)), nrow=n, ncol=p)
beta <- matrix(rnorm(p) * rbinom(p, 1, 0.2), ncol=1)
y <- config$rng(config$link(X %*% beta))

fitted <- glmnet::glmnet(X, y, family = family, alpha=0.9)

out_file <- file(out_name, "wb")

writeBin(as.integer(n), out_file, size=4)
writeBin(as.integer(p), out_file, size=4)
writeBin(length(fitted$lambda), out_file, size=4)
writeBin(as.vector(X), out_file)
writeBin(as.vector(as.matrix(fitted$beta)), out_file)
writeBin(as.vector(y), out_file)
writeBin(fitted$lambda, out_file)

close(out_file)
