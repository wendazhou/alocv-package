#! /usr/env Rscript --no-restore --no-save

library(glmnet)

args <- commandArgs(TRUE)

if(length(args) != 3) {
    print("Expected three arguments")
    quit(status = 1)
}

n <- as.integer(args[1])
p <- as.integer(args[2])
out_name <- args[3]

X <- matrix(rnorm(n * p), nrow=n, ncol=p)
beta <- matrix(rnorm(p) * rbinom(p, 1, 0.2), ncol=1)
y <- X %*% beta + rnorm(n, sd=0.1)

fitted <- glmnet(X, y)

out_file <- file(out_name, "wb")

writeBin(as.integer(n), out_file, size=4)
writeBin(as.integer(p), out_file, size=4)
writeBin(length(fitted$lambda), out_file, size=4)
writeBin(as.vector(X), out_file)
writeBin(as.vector(as.matrix(fitted$beta)), out_file)
writeBin(as.vector(y), out_file)
writeBin(fitted$lambda, out_file)

close(out_file)
