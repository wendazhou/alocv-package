source('./generation_utilities.R')

svm.alo <- Rcpp::cppFunction('SEXP svmKerALO(
    const arma::mat &K, const arma::vec &y, const arma::vec &alpha,
    const double &rho, const double &lambda, const double &tol) {
  arma::uword N = y.n_elem;

  // augment the data and weight matrices with offset
  arma::mat Kinv = arma::inv(K); // cannot use inv_sympd because of sigmoid kernel

  arma::vec yHat = K * alpha - rho;
  arma::vec yyHat = y % yHat;

  // identify singularities
  arma::uvec V = arma::find(arma::abs(1 - yyHat) < tol);
  arma::uvec S = arma::find(arma::abs(1 - yyHat) >= tol);

  // useful matrices
  arma::mat I_n = arma::eye<arma::mat>(N, N);
  arma::mat KV = K.cols(V);
  arma::mat KS = K.cols(S);
  arma::mat K1 = arma::inv(KV.t() * Kinv * KV);

  arma::uvec gID = arma::intersect(arma::find(yyHat < 1.0), S);
  arma::mat yK_g = K.cols(gID);
  yK_g.each_row() %= arma::trans(y.elem(gID));

  // containers for a and g
  arma::vec a = arma::zeros<arma::vec>(N);
  arma::vec g = arma::zeros<arma::vec>(N);

  // compute a and g for S
  arma::mat KT = Kinv * KV * K1 * KV.t() * Kinv;
  arma::mat KO = Kinv * (I_n - KV * K1 * KV.t() * Kinv);
  arma::mat Ka_s = KS.t() * KO * KS / lambda;
  a.elem(S) = arma::diagvec(Ka_s);
  g.elem(gID) = -y.elem(gID);

  // compute a and g for V
  arma::vec gradR = lambda * K * alpha;
  arma::vec sum_yK = arma::sum(yK_g, 1);
  a.elem(V) = 1 / (lambda * arma::diagvec(K1));
  g.elem(V) = arma::inv(KV.t() * KV) * KV.t() * (sum_yK - gradR);

  arma::vec yalo = yHat + a % g;
  return Rcpp::List::create(
    Rcpp::Named("yalo") = Rcpp::wrap(yalo),
    Rcpp::Named("a") = Rcpp::wrap(a),
    Rcpp::Named("g") = Rcpp::wrap(g),
    Rcpp::Named("K1") = Rcpp::wrap(K1),
    Rcpp::Named("KV") = Rcpp::wrap(KV),
    Rcpp::Named("KS") = Rcpp::wrap(KS),
    Rcpp::Named("KO") = Rcpp::wrap(KO),
    Rcpp::Named("KT") = Rcpp::wrap(KT),
    Rcpp::Named("Kinv") = Rcpp::wrap(Kinv),
    Rcpp::Named("V") = Rcpp::wrap(V),
    Rcpp::Named("S") = Rcpp::wrap(S),
    Rcpp::Named("gS") = Rcpp::wrap(gID),
    Rcpp::Named("syK") = Rcpp::wrap(sum_yK));
}', depends='RcppArmadillo')

polyKer <- Rcpp::cppFunction(
  'arma::mat polynomialKer(const arma::mat &X, const double &gamma, const double &coef0, const int &degree) {
    // compute the kernel matrix
    arma::mat K = arma::pow(gamma * X * X.t() + coef0, degree);

    return K;
  }', depends='RcppArmadillo')

radialKer <- function(X, gamma) {
  exp(-gamma * as.matrix(dist(X)) ^ 2)
}

n <- 200
p <- 50
n2 <- round(n / 2)

set.seed(42)

X <- matrix(rnorm(n / 2 * p), nrow=n, ncol=p)
X[1:n2,] <- X[1:n2,] + 2.5
X[(n2+1):n,] <- X[(n2+1):n,] - 2.5
g <- 2 / p
degree <- 2.5
lambda <- 1

K <- radialKer(X, g)

y <- c(rep(1, n/2), rep(-1, n/2 - 1), 1)

svm.fit <- e1071::svm(y ~ X, scale = T, kernel='radial', degree=degree, gamma=g, cost=1 / lambda,
                      tolerance=1e-6, type='C-classification')

alpha <- rep(0, n)
alpha[svm.fit$index] <- svm.fit$coefs

yalo_info <- svm.alo(K, y, alpha, svm.fit$rho, lambda, 1e-5)
yalo <- yalo_info[['yalo']]
mean_hinge <- mean(pmax(0, 1 - yalo * y))


text <- values_to_c(
    n = as.integer(n),
    p = as.integer(p),
    gamma = g,
    expected_hinge = mean_hinge,
    rho = svm.fit$rho,
    lambda = lambda,
    K = K,
    alpha = alpha,
    y = y,
    yalo = yalo)

f <- file('../examples/svm_example.in')
writeLines(text, f)
close(f)
