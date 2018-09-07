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
                   NumericVector lambda, double alpha = 1.0,
                   bool has_intercept = true, Nullable<NumericVector> a0 = R_NilValue,
                   double tolerance = 1e-5, bool use_rfp = true) {
    NumericVector alo(B.ncol());
    NumericMatrix leverage(A.nrow(), B.ncol());

    enet_compute_alo_d(A.nrow(), A.ncol(), B.ncol(), &A[0], A.nrow(),
                       &B[0], B.nrow(), &y[0],
                       a0.isNull() ? nullptr : &a0.as()[0],
                       &lambda[0], alpha, has_intercept, use_rfp, tolerance,
                       &alo[0], &leverage[0]);

    List result;
    result["alo"] = alo;
    result["leverage"] = leverage;
    return result;
}
