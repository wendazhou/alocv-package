#include <Rcpp.h>
using namespace Rcpp;

#include "alocv/alo_lasso.h"


// [[Rcpp::export]]
List alo_lasso_rcpp(NumericMatrix A, NumericMatrix B, NumericVector y) {
    NumericVector alo(B.ncol());

    lasso_compute_alo_d(A.nrow(), A.ncol(), B.ncol(), &A[0], 1,
                        &B[0], 1, &y[0], 1, 1e-5, &alo[0], nullptr);
}


// [[Rcpp::export]]
NumericVector timesTwo(NumericVector x) {
  return x * 2;
}
