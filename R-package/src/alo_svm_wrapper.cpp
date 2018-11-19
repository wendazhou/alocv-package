#include <Rcpp.h>
using namespace Rcpp;

#include "alocv/alo_svm.h"

// [[Rcpp::export]]
List alo_svm_rcpp(NumericMatrix K, NumericVector y, NumericVector alpha, double rho, double lambda, double tolerance = 1e-5) {
    NumericMatrix k_copy(Rcpp::clone(K));
    double alo_hinge_loss;

    if(K.nrow() != K.ncol()) {
        Rcpp::stop("K must be a square matrix.");
    }

    if(K.nrow() != y.size()) {
        Rcpp::stop("Incompatible length of observations y.");
    }

    if(K.nrow() != alpha.size()) {
        Rcpp::stop("Incompatible length of dual fitted values alpha.");
    }

    NumericVector alo_predicted(K.nrow());

    svm_compute_alo(K.nrow(), &k_copy[0], &y[0], &alpha[0], rho, lambda, tolerance,
                    &alo_predicted[0], &alo_hinge_loss);

    return Rcpp::List::create(
        Rcpp::Named("predicted") = alo_predicted,
        Rcpp::Named("loss") = alo_hinge_loss
    );
}
