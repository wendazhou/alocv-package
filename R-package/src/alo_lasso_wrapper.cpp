#include <Rcpp.h>
using namespace Rcpp;

#include "alocv/alo_lasso.h"
#include "alocv/alo_enet.h"


// [[Rcpp::export]]
List alo_lasso_rcpp(NumericMatrix A, NumericMatrix B, NumericVector y) {
    NumericVector alo(B.ncol());
    NumericMatrix leverage(A.nrow(), B.ncol());

    lasso_compute_alo_d(A.nrow(), A.ncol(), B.ncol(), &A[0], A.nrow(),
                        &B[0], B.nrow(), &y[0], 1, 1e-5, &alo[0], &leverage[0]);

    List result;
    result["alo"] = alo;
    result["leverage"] = leverage;
    return result;
}

// [[Rcpp::export]]
List alo_enet_rcpp(NumericMatrix A, NumericMatrix B, NumericVector y,
                   NumericVector lambda, double alpha, bool has_intercept) {
    NumericVector alo(B.ncol());
    NumericMatrix leverage(A.nrow(), B.ncol());

    enet_compute_alo_d(A.nrow(), A.ncol(), B.ncol(), &A[0], A.nrow(),
                       &B[0], B.nrow(), &y[0], &lambda[0],
                       alpha, has_intercept, 1e-5,
                       &alo[0], &leverage[0]);

    List result;
    result["alo"] = alo;
    result["leverage"] = leverage;
    return result;
}
