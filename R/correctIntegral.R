#' @title Normalizes Log-Concave Density
#'
#' @description \code{correctIntegral} normalizes a log-concave density
#'   parametrized by a set of hyperplane parameters. Works by calculating y =
#'   log(f(x)) for each data point in X, normalizing y, and then recalculating a
#'   and b.
#'
#' @param X Set of data points (one sample per row)
#' @param mu Mean vector of X that gets added back to X
#' @param a Matrix where rows are slopes of hyperplanes
#' @param b Vector where entries are offsets of hyperplanes
#' @param cvh Matrix where each row is a set of indices of points in X
#'   describing one face of conv(X)
#'
#' @return Normalized hyperplane parameters (for the uncentered \code{X <- X +
#'   mu}) \item{aOpt, bOpt}{Hyperplane parameters of the normalized density.}
#'   \item{y}{Vector with values y_i = log(f(X_)) of the normalized density.}
#'   \item{aOptSparse, bOptSparse}{Input hyperplane parameters.}
#'
#' @example Examples/correctIntegral

correctIntegral <- function(X, mu, a, b, cvh) {
  n <- dim(X)[1]
  d <- dim(X)[2]

  XTmp <- matrix(runif(length(b) * (d+1) * d), length(b) * (d+1), d)
  yTmp <- apply(do.call(rbind, replicate((d+1), -a, simplify = FALSE)) * XTmp, 1, sum) - rep(b, d+1)
  # add mean
  XTmp <- XTmp + t(matrix(rep(mu, length(b)*(d+1)), d, length(b) * (d+1)))
  T = matrix(1:(length(b) * (d+1)), length(b), d+1)
  # calc new params
  r <- .C(  "recalcParamsC",
            as.double(t(XTmp)),
            as.double(yTmp),
            as.integer(t(T-1)),
            as.integer(dim(T)[1]),
            as.integer(d),
            a = as.double(matrix(0,dim(T)[1] * d)),
            b = as.double(matrix(0,dim(T)[1])))

  # sparse set of hyperplanes corrected for the mean
  aOptSparse <- matrix(r$a, length(r$b), d)
  bOptSparse <- r$b

  y = apply(-a %*% t(X) - matrix(rep(b, n), length(b), n), 2, min)
  idxCVH <- unique(as.vector(cvh))
  P <- matrix(c(X, t(y)), nrow = n)
  Q <- matrix(c(X[idxCVH, ], rep(min(y[idxCVH]) - 1, length(idxCVH))), nrow = length(idxCVH))

  X <- X + t(matrix(rep(mu,n), 2, n))
  r <- callCalcExactIntegralC(X,y,P,Q,1e-10)

  aOptNew <- t(matrix(r$a, d, length(r$a) / d))
  bOptNew <- r$b

  yEval <- aOptNew %*% t(X[1:min(n,10), ]) + matrix(rep(bOptNew, min(10, n)), length(bOptNew), min(10, n))
  diff <- apply(yEval, 2, max) + r$y[1:min(10, length(y))]
  if (sqrt(sum(diff^2)) > 1e-6) {
    warning('Potential numerical problems when calculating the final set of hyperplanes --> Recommended to run the optimization again')
  }

  return(list("a" = aOptNew, "b" = bOptNew, "aSparse" = aOptSparse, "bSparse" = bOptSparse, "logMLE" = r$y))
}
