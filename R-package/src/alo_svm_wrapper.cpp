#include <Rcpp.h>
using namespace Rcpp;

#include "alocv/alo_svm.h"
#include <memory>

enum class KernelType : int {
    Radial = 0,
    Polynomial = 1,
    Linear = 2,
};


blas_size num_elements_and_check(const NumericMatrix& K, const NumericVector& y,
                                 const NumericVector& alpha, bool use_rfp) {
    blas_size num_elements;
    blas_size n = static_cast<blas_size>(y.size());

    if(use_rfp) {
        blas_size ldk = (n % 2 == 1) ? n : n + 1;
        num_elements = n * (n + 1) / 2;

        if(ldk != K.nrow() || ldk * K.ncol() != num_elements) {
            Rcpp::stop("Shape of K is not compatible with observation y");
        }
    }
    else {
        if(K.nrow() != n || K.ncol() != n) {
            Rcpp::stop("K must be a square matrix of size n x n.");
        }

        num_elements = n * n;
    }

    if(n != y.size()) {
        Rcpp::stop("Incompatible length of observations y.");
    }

    if(n != alpha.size()) {
        Rcpp::stop("Incompatible length of dual fitted values alpha.");
    }

    return num_elements;
}


// [[Rcpp::export]]
List alo_svm_kernel_rcpp(NumericMatrix K, NumericVector y, NumericVector alpha,
                  double rho, double lambda, double tolerance = 1e-5,
                  bool use_rfp = false) {
    blas_size n = y.size();
    auto num_elements = num_elements_and_check(K, y, alpha, use_rfp);

    auto k_copy = std::unique_ptr<double[]>(new double[num_elements]);
    std::copy(&K(0, 0), &K(0, 0) + num_elements, k_copy.get());

    NumericVector alo_predicted(n);
    double alo_hinge_loss;

    svm_compute_alo(n, k_copy.get(), &y[0], &alpha[0], rho, lambda, tolerance,
                    &alo_predicted[0], &alo_hinge_loss, use_rfp);

    return Rcpp::List::create(
        Rcpp::Named("predicted") = alo_predicted,
        Rcpp::Named("loss") = alo_hinge_loss
    );
}

void compute_kernel_impl(blas_size n, blas_size p, double* X, double* K,
                         int kernel_type, double gamma, double degree, double coef0,
                         bool use_rfp=false) {
    switch(static_cast<KernelType>(kernel_type)) {
    case KernelType::Radial:
        svm_kernel_radial(n, p, X, gamma, K, use_rfp);
        return;
    case KernelType::Polynomial:
        if (degree < 0) {
            Rcpp::stop("Degree of polynomial kernel < 0!");
        }
        svm_kernel_polynomial(n, p, X, K, gamma, degree, coef0, use_rfp);
        return;
    default:
        stop("Unknown kernel type.");
    }
}

// [[Rcpp::export]]
NumericMatrix compute_svm_kernel(NumericMatrix X, int kernel_type,
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

    compute_kernel_impl(n, X.ncol(), &X(0, 0), &output(0, 0),
        kernel_type, gamma, degree, coef0, use_rfp);

    return output;
}

// [[Rcpp::export]]
List alo_svm_rcpp(
        NumericMatrix X, NumericVector y, NumericVector alpha,
        double rho, double lambda, int kernel_type,
        double gamma, double degree, double coef0,
        double tolerance = 1e-5, bool use_rfp = false) {

    blas_size n = y.size();

    if (X.nrow() != n) {
        Rcpp::stop("Feature matrix X and predictor matrix y are not compatible.");
    }

    if (n != alpha.size()) {
        Rcpp::stop("Fitted vector alpha is not of the expected size.");
    }

    auto num_elements = use_rfp ? n * (n + 1) / 2 : n * n;
    auto K = std::unique_ptr<double[]>(new double[num_elements]);
    std::fill(K.get(), K.get() + num_elements, 0.0);

    compute_kernel_impl(n, X.ncol(), &X(0, 0), K.get(),
        kernel_type, gamma, degree, coef0, use_rfp);

    NumericVector alo_predicted(n);
    double alo_hinge_loss;

    svm_compute_alo(n, K.get(), &y[0], &alpha[0], rho, lambda, tolerance,
        &alo_predicted[0], &alo_hinge_loss, use_rfp);

    return Rcpp::List::create(
        Rcpp::Named("predicted") = alo_predicted,
        Rcpp::Named("loss") = alo_hinge_loss
    );
}
