#' @title Optimize MLE objective for a log-concave density
#'
#' @description \code{callNewtonBFGSLC} executes the quasi-Newton
#'   optimization of the maximum likelihood estimator for a log-concave
#'   density f(x). Calls a underlying C function.
#'
#' @param X Set of data points (one sample per row)
#' @param w Vector of sample weights
#' @param params Vector of initial hyperplane parameters
#' @param paramsKernel Alternative vector of initial hyperplane parameters
#' @param cvhParams List that contains parametrization of the
#'   convex hull of X in terms of its faces
#' @param gamma Parameters that governs the smoothness of the log-concave
#'   density (default: 1000)
#' @param verbose The amount of information printed during the optimization;
#'   valid levels are {0,1,2} (default: 0)
#' @param intEps Threshold of the numerical integration error below which the
#'   optimization may terminite (default: 1e-3)
#' @param objEps Threshold of the size of the function step taken below which
#'   the optimziation may terminite (default: 1e-7)
#' @param offset Threshold that governs how fast inactive hyperplanes are dropped;
#'   smaller values correspond to a slower rate of hyperplane deletion
#'   (default: 1e-1)
#' @param maxIter Maximual number of iterations of Newton optimization (default: 1e4)
#'
#' @return List of n hyperplanes that describe the upper convex hull of log(f(x))
#'   \item{a}{Slopes of hyerplanes (n x d matrix)}
#'   \item{b}{Offsets of hyperplanes}

callNewtonBFGSLC <- function (X, w, params, paramsKernel, cvhParams, gamma=1000, verbose=0, intEps = 1e-3, objEps = 1e-7, offset = 1e-1, maxIter = 1e4) {
  n <- dim(X)[1]
  d <- dim(X)[2]

  r <- .C(  "newtonBFGSLC",
              as.double(X),
              as.double(w),
              as.double(c(apply(X, 2, min), apply(X, 2, max))),
              params = as.double(params),
              as.double(paramsKernel),
              lenP = as.integer(length(params)),
              as.integer(length(paramsKernel)),
              as.integer(d),
              as.integer(n),
              as.double(cvhParams$ACVH),
              as.double(cvhParams$bCVH),
              as.integer(length(cvhParams$bCVH)),
              as.double(intEps),
              as.double(objEps),
              as.double(offset),
              as.integer(verbose),
              as.double(gamma),
              as.integer(maxIter))
  # res$lenP denotes the number of active parameters in res$params
  optParams <- r$params[1:r$lenP]

  nH <- r$lenP/(d+1) # number of hyperplanes
  a <- matrix(optParams[1:(d * nH)],nH,d)
  b <- optParams[(d*nH+1):length(optParams)]
  return(list("a" = a, "b" = b))
}

#' @title Optimize MLE objective for smooth a log-concave density
#'
#' @description \code{callNewtonBFGSLInitC} is similar as \code{callNewtonBFGSLC}
#'   but optimizies a smooth log-concave density f(x) with parameter gamma=1. Due to
#'   numerical reasons a different set of C functions is used.
#'
#' @param X Set of data points (one sample per row)
#' @param w Vector of sample weights
#' @param params Vector of initial hyperplane parameters
#' @inheritParams paramFitGammaOne
#'
#' @return List containing the optimal parameters as well as the log likelihood
#'   \item{params}{Stacked vector of hyperplane slopes and offsets}
#'   \item{logLike}{Vector containing function evaluations log(f(x_i))}

callNewtonBFGSLInitC <- function (X, w, params, ACVH, bCVH) {
  n <- dim(X)[1]
  d <- dim(X)[2]

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
  return(r)
}

#' @title Analytically normalizes density to one
#'
#' @description \code{callCalcExactIntegralC} is a wrapper to a C function that
#'   analytically normalizes a log-concave density, uniquely described by X and y,
#'   where y_i = log(f(X_i)), by adapting the function values y uniformly by some delta.
#'
#' @param X Set of data points (one sample per row)
#' @param y Vector of function evaluations log(f(X_i))
#' @param filter Vector of with entries TRUE/FALSE indicating
#'   which data points to discard (indicated by FALSE)
#' @param eps The maximum integration error allowed
#' @inheritParams paramFitGammaOne
#'
#' @return List containing the normalized parametrers
#'   \item{a}{Vector of hyperplane slopes}
#'   \item{b}{Vector of hyperplane offsets}

callCalcExactIntegralC <- function(X, y, cvh, filter, eps) {

  idxCVH <- setdiff(unique(as.vector(cvh)), which(!filter))
  P <- matrix(c(X[filter, ], y[filter]), nrow = length(filter))
  Q <- matrix(c(X[idxCVH, ], rep(min(y[idxCVH]) - 1, length(idxCVH))), nrow = length(idxCVH))

  X = X[filter, ]
  y = y[filter]

  n <- dim(X)[1]
  d <- dim(X)[2]

  T <- geometry::convhulln(rbind(P, Q))
  T <- T[!(apply(T, 1, max) > n), ]

  # if there is only one simplex
  if (class(T) != "matrix") {
    T <- t(as.matrix(T))
  }

  r <- .C(  "calcExactIntegralC",
          as.double(t(X)),
          y = as.double(y),
          as.integer(t(T-1)),
          as.integer(dim(T)[1]),
          as.integer(n),
          as.integer(d),
          as.double(1),
          as.double(eps),
          a = as.double(matrix(0, dim(T)[1] * d)),
          b = as.double(matrix(0, dim(T)[1])))
  r$a = t(matrix(r$a, d, length(r$a) / d))
  return(r)
}


#' @title Checks if AVX is active
#'
#' @description \code{compilationInfo} is a wrapper to a C function, which prints information
#'   about whether AVX extensions where activated during compilation. Active AVX speeds up
#'   computations significantly (by a factor of 4 to 8 roughly). More informations about how
#'   AVX can be enabled during installation can be found in documentation.

compilationInfo <- function() {
  invisible(.C ("printAVXInfo"))
}
