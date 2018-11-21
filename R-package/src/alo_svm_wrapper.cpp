#include <Rcpp.h>
using namespace Rcpp;

#include "alocv/alo_svm.h"

enum class KernelType : int {
    Radial = 0,
    Polynomial = 1,
    Linear = 2,
};


// [[Rcpp::export]]
List alo_svm_rcpp(NumericMatrix K, NumericVector y, NumericVector alpha,
                  double rho, double lambda, double tolerance = 1e-5,
                  bool use_rfp = false) {
    double alo_hinge_loss;
    blas_size n = y.size();

    double* k_copy;

    if(use_rfp) {
        blas_size ldk = (n % 2 == 1) ? n : n + 1;
        blas_size n_elements = n * (n + 1) / 2;

        if(ldk != K.nrow() || ldk * K.ncol() != n_elements) {
            Rcpp::stop("Shape of K is not compatible with observation y");
        }

        k_copy = new double[n_elements];
        std::copy(&K(0, 0), &K(0, 0) + n_elements, k_copy);
    }
    else {
        if(K.nrow() != n || K.ncol() != n) {
            Rcpp::stop("K must be a square matrix of size n x n.");
        }
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
                             double gamma, double degree, double coef0,
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
    case KernelType::Polynomial:
        if (degree < 0) {
            Rcpp::stop("Degree of polynomial kernel < 0!");
        }
        svm_kernel_polynomial(n, X.ncol(), &X(0, 0), &output(0, 0), gamma, degree, coef0, use_rfp);
        return output;
    default:
        stop("Unknown kernel type.");
    }
}
