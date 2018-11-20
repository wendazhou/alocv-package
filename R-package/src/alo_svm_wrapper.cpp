#include <Rcpp.h>
using namespace Rcpp;

#include "alocv/alo_svm.h"

enum class KernelType : int {
    Radial = 0,
    Polynomial = 1,
    Linear = 2,
};

blas_size RfpGetSize(NumericMatrix const& K) {
    return static_cast<blas_size>((K.ncol() + K.nrow() * 2 + 1) / 2);
}

// [[Rcpp::export]]
List alo_svm_rcpp(NumericMatrix K, NumericVector y, NumericVector alpha,
                  double rho, double lambda, double tolerance = 1e-5,
                  bool use_rfp = false) {
    double alo_hinge_loss;
    blas_size n;

    double* k_copy;

    if(use_rfp) {
        if(std::abs(K.nrow() - K.ncol() * 2) > 1) {
            Rcpp::stop("Invalid size for K in RFP format");
        }
        n = RfpGetSize(K);
        k_copy = new double[n * (n + 1) / 2];
        std::copy(&K(0, 0), &K(0, 0) + n * (n + 1) / 2, k_copy);
    }
    else {
        if(K.nrow() != K.ncol()) {
            Rcpp::stop("K must be a square matrix.");
        }
        n = static_cast<blas_size>(K.ncol());
        k_copy = new double[n * n];
        std::copy(&K(0, 0), &K(0, 0) + n * n, k_copy);
    }

    if(n != y.size()) {
        Rcpp::stop("Incompatible length of observations y.");
    }

    if(n != alpha.size()) {
        Rcpp::stop("Incompatible length of dual fitted values alpha.");
    }

    NumericVector alo_predicted(n);

    svm_compute_alo(n, k_copy, &y[0], &alpha[0], rho, lambda, tolerance,
                    &alo_predicted[0], &alo_hinge_loss, use_rfp);

    delete k_copy;

    return Rcpp::List::create(
        Rcpp::Named("predicted") = alo_predicted,
        Rcpp::Named("loss") = alo_hinge_loss
    );
}

// [[Rcpp::export]]
NumericMatrix alo_svm_kernel(NumericMatrix X, int kernel_type,
                             double gamma, int degree, double coef0,
                             bool use_rfp=false) {
    NumericMatrix output;

    auto n = X.nrow();
    bool is_odd = n % 2 == 1;

    if(use_rfp) {
        output = NumericMatrix(n + (is_odd ? 0 : 1), (n + 1) / 2);
    }
    else {
        output = NumericMatrix(n, n);
    }

    switch(static_cast<KernelType>(kernel_type)) {
    case KernelType::Radial:
        svm_kernel_radial(n, X.ncol(), &X(0, 0), gamma, &output(0, 0), use_rfp);
        return output;
    default:
        stop("Unknown kernel type.");
    }
}
