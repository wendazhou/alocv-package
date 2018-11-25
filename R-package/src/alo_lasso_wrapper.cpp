#include <Rcpp.h>
using namespace Rcpp;

#include "alocv/alo_lasso.h"
#include "alocv/alo_enet.h"


//' Computes alo values for the LASSO problem using a up/down-dating
//' algorithm to reuse intermediate values along a given regularization
//' path.
//'
//' Note that due to the special algorithm, it is not strictly correct
//' for models fitted with an intercept, as it does not account for
//' the uncertaining in estimating the intercept.
//'
//' @param A The predictor matrix (\code{n x p}).
//' @param B The matrix of fitted values for each regularization (\code{p x m})
//' @param y The vector of observed responses (length \code{n}).
//' @param a0 An optional vector of fitted intercepts.
//'
//' @return A list with the computed alo values and leverage.
//'
// [[Rcpp::export]]
List alo_lasso_rcpp(NumericMatrix A, NumericMatrix B,
                    NumericVector y, Nullable<NumericVector> a0 = R_NilValue) {
    NumericVector alo(B.ncol());
    NumericMatrix leverage(A.nrow(), B.ncol());

    lasso_compute_alo_d(A.nrow(), A.ncol(), B.ncol(), &A(0, 0), A.nrow(),
                        &B(0, 0), B.nrow(), &y(0), 1,
                        a0.isNull() ? nullptr : &a0.as()[0],
                        1e-5, &alo[0], &leverage[0]);

    return Rcpp::List::create(
		Rcpp::Named("alo") = alo,
		Rcpp::Named("leverage") = leverage
    );
}


//' Computes alo values for the elastic-net problem (and its GLM forms).
//' Note that this function is significantly slower than \code{\link{alo_lasso_rcpp}}
//' for large problems, as it must solve each problem separately.
//'
//' @param A The predictor matrix
//' @param B The matrix of estimated coefficients for each tuning value.
//' @param y The vector of observed responses.
//' @param lambda The vector of tuning values.
//' @param family An enumeration indicating the GLM family.
//' @param alpha The elastic-net parameter (balance between l1 and l2 penalty).
//' @param a0 If not NULL, the estimated intercept values.
//' @param tolerance The numerical tolerance for detecting active sets.
//' @param use_rfp If true, uses a more compact storage format for symmetric matrices.
//'
// [[Rcpp::export]]
List alo_enet_rcpp(NumericMatrix A, NumericMatrix B, NumericVector y,
                   NumericVector lambda, int family = 0, double alpha = 1.0,
                   Nullable<NumericVector> a0 = R_NilValue,
                   double tolerance = 1e-5, bool use_rfp = true) {
    NumericVector alo(B.ncol());
    NumericVector alo_mse(B.ncol());
    NumericVector alo_mae(B.ncol());
    NumericMatrix leverage(A.nrow(), B.ncol());

    enet_compute_alo_d(A.nrow(), A.ncol(), B.ncol(), &A(0, 0), A.nrow(),
                       &B(0, 0), B.nrow(), &y[0],
                       a0.isNull() ? nullptr : &a0.as()[0],
                       &lambda[0], alpha, a0.isNotNull(), (GlmFamily)family,
                       use_rfp, tolerance,
                       &alo[0], &leverage[0], &alo_mse[0], &alo_mae[0]);

	return Rcpp::List::create(
		Rcpp::Named("alo") = alo,
		Rcpp::Named("leverage") = leverage,
		Rcpp::Named("alo_mse") = alo_mse,
		Rcpp::Named("alo_mae") = alo_mae
	);
}
