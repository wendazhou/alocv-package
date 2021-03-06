% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{alo_enet_rcpp}
\alias{alo_enet_rcpp}
\title{Computes alo values for the elastic-net problem (and its GLM forms).
Note that this function is significantly slower than \code{\link{alo_lasso_rcpp}}
for large problems, as it must solve each problem separately.}
\usage{
alo_enet_rcpp(
  A,
  B,
  y,
  lambda,
  family = 0L,
  alpha = 1,
  a0 = NULL,
  tolerance = 1e-05,
  use_rfp = TRUE
)
}
\arguments{
\item{A}{The predictor matrix}

\item{B}{The matrix of estimated coefficients for each tuning value.}

\item{y}{The vector of observed responses.}

\item{lambda}{The vector of tuning values.}

\item{family}{An enumeration indicating the GLM family.}

\item{alpha}{The elastic-net parameter (balance between l1 and l2 penalty).}

\item{a0}{If not NULL, the estimated intercept values.}

\item{tolerance}{The numerical tolerance for detecting active sets.}

\item{use_rfp}{If true, uses a more compact storage format for symmetric matrices.}
}
\description{
Computes alo values for the elastic-net problem (and its GLM forms).
Note that this function is significantly slower than \code{\link{alo_lasso_rcpp}}
for large problems, as it must solve each problem separately.
}
