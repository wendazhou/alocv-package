## ----setup, include = FALSE----------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 6, 
  fig.height = 4
)

## ---- eval = TRUE, echo = TRUE-------------------------------------------
library(alocv)
library(glmnet)

## ---- eval = TRUE--------------------------------------------------------
n = 500
p = 200
k = 100
beta = rnorm(p, 0, 1)
beta[-(1:k)] = 0

X = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
y = X %*% beta + rnorm(n, 0, 0.5)
y[y >= 0] = 2 * sqrt(y[y >= 0])
y[y < 0] = -2 * sqrt(-y[y < 0])

## ---- eval = TRUE--------------------------------------------------------
ptm = proc.time()
CV_el = cv.glmnet(X, y, alpha = 0.5, nfolds = n, 
                  grouped = F, intercept = T, standardize = T, type.measure = "mse")
proc.time() - ptm

## ---- eval = TRUE--------------------------------------------------------
ptm = proc.time()
GLM_el = glmnet(X, y, lambda = CV_el$lambda, alpha = 0.5, intercept = T, standardize = T)
ALO_el = alo_glmnet(X, y, alpha=0.5, standardize=T, intercept=T)
proc.time() - ptm

## ------------------------------------------------------------------------
plot(CV_el$cvm, xlab = "lambda", ylab = "Risk", type = "l", lwd = 2, col = "darkorange")
lines(ALO_el$alo, type = "b", pch = 4, lwd = 2, col = 4)

