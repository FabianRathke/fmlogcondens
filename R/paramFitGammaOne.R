#' @title Parameter Initialization Based on a Smooth Log-Concave Density
#'
#' @description \code{paramFitGammaOne} fits in a first step a log-concave
#'   density to data points X with weights w using a smoothness parameter of
#'   gamma=1. In a second step, it calculates the upper convex hull of the set X
#'   and log(y), where y_i is the smooth log-concave density evaluated at X_i.
#'   It returns the hyperplane parameters of the faces of this upper convex
#'   hull.
#'
#' @param X Set of data points (one sample per row)
#' @param w Vector with weights for X (\code{sum(w)==1})
#' @param ACVH Matrix where each row constitutes the normal vector of a face of
#'   conv(X)
#' @param bCVH Vector where each entry constitutes the offset for a face of
#'   conv(X)
#' @param cvh Matrix where each row is a set of indices of points in X
#'   describing one face of conv(X)
#'
#' @return A list containing the description of the upper convex hull of
#'   (X,log(y)) in term of hyperplane parameters: \item{a}{A matrix where each
#'   row constitutes the normal vector of a face} \item{b}{A vector where each
#'   entry constitutes the offset of a face}
#'
#' @example Examples/paramFitGammaOne


paramFitGammaOne <- function(X, w, ACVH, bCVH, cvh) {
  n <- dim(X)[1]
  d <- dim(X)[2]
  m <- 10 * d # number of initial hyperplanes

  minLogLike <- 1e5
  numInits <- 1
  # choose best model from a set of initializations
  for (i in 1:numInits) {
    a <- runif(m * d, 0, 0.1)
    b <- runif(m)
    params <- c(a, b)
    r <- .C("newtonBFGSLInitC",
              as.double(X),
              as.double(w),
              as.double(c(apply(X, 2, min),apply(X, 2, max))),
              params = as.double(params),
              as.integer(d),
              as.integer(length(params)),
              as.integer(n),
              as.double(ACVH),
              as.double(bCVH),
              as.integer(length(bCVH)),
              as.double(1e-3),
              as.double(1e-6),
              logLike = as.double(matrix(0, 1)))
    if (r$logLike < minLogLike) {
      optParams <- r$params
      minLogLike <- r$logLike
    }
  }

  aOpt <- matrix(optParams[1:(m * d)], m, d)
  bOpt <- rep(optParams[(m * d + 1):length(optParams)])

  yT <- matrix(-log(apply(exp(aOpt %*% t(X) + matrix(rep(bOpt, n), length(bOpt), n)), 2, sum)), n, 1)
  idxCVH <- unique(as.vector(cvh))
  P <- matrix(c(X, yT), nrow = n)
  Q <- matrix(c(X[idxCVH, ], rep(min(yT[idxCVH]) - 1, length(idxCVH))), nrow = length(idxCVH))
  T <- geometry::convhulln(rbind(P, Q))
  T <- T[!(apply(T, 1, max) > n), ]

  nH = dim(T)[1]
  # normalize density to one for exact integral
  r <- .C("calcExactIntegralC",
            as.double(t(X)),
            as.double(yT),
            as.integer(t(T-1)),
            as.integer(nH),
            as.integer(n),
            as.integer(d),
            as.double(1),
            as.double(0.01),
            a = as.double(matrix(0,nH * d)),
            b = as.double(matrix(0,nH)))

  aOpt <-t(matrix(r$a, d, length(r$a) / d))
  bOpt <- r$b

  # limit the number of hyperplanes for 1D to 1000
  if (d == 1 && length(bOpt) > 1000) {
    idxSelect <- sample(length(bOpt), 1000)
    aOpt <- aOpt[idxSelect]
    bOpt <- bOpt[idxSelect]
  }

  # detect if all hyerplanes where intitialized to the same parameters; happens for small sample sizes --> very slow convergence in the final optimization
  idxCheck <- sample(length(bOpt), min(100, length(bOpt)))
  if (mean(apply(aOpt[idxCheck, ], 2, var)) < 1e-4 || length(bOpt) == 1) {
    print('#### Bad initialization due to small sample size, switch to kernel kensity based initialization ####');
    params <- paramFitKernelDensity(X, w, cvh)
  } else {
    params <- list("a" = aOpt, "b" = bOpt)
  }

  return(params)
}