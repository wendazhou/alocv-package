library(glmnet)

n <- 1000
p <- 1600

x <- matrix(rnorm(n * p), nrow = n, ncol = p)
beta <- matrix(rnorm(p) * rbinom(p, 1, 0.5), ncol=1)
y <- x %*% beta + rnorm(n, sd = 0.1)

s0 <- proc.time()
fitted <- glmnet(x, y, standardize = F, intercept = F, alpha=0.5)
t0 <- proc.time() - s0

s2 <- proc.time()
alo2 <- alocv2::glmnetALO(x, y, fitted, 0.5, standardize = F)
t2 <- proc.time() - s2

s1 <- proc.time()
sy <- mean((y - mean(y)) ^ 2)
alo1 <- alocv:::alo_enet_rcpp(x, as.matrix(fitted$beta), y,
                             lambda = fitted$lambda * (n / sy),
                             alpha = 0.5, has_intercept = F,
                             use_rfp = T)
t1 <- proc.time() - s1


list(fit = t0, mine = t1, intern = t2)
