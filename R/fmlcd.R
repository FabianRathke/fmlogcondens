#' @title Estimates a Log-Concave Density
#'
#' @description \code{fmlcd} returns a MLE estimate of a log-concave
#'   density for \code{X}. After obtaining an initial parameter estimate the MLE
#'   objective with log-concavity and normalization constraint is optimized
#'   using a quasi-Newton approach for large scale optimization (BFGS-L). The
#'   logarithm of the optimal density f(x) is a piecewise-linear function. Its
#'   parametrization in terms of a set of hyperplanes is returned.
#'
#' @param X Matrix of data points (one sample per row)
#' @param w Vector of sample weights
#' @param init String determining the initialization approach ('kernel' based on
#'   kernel density, 'smooth' based on smooth log-concave density, '' compares
#'   both and takes the optimal one)
#' @param verbose Int determining the verboseness of the code (0 (no output)
#'   to 3)
#'
#' @return Parametrization of f(x) in terms of hyperplanes and function
#'   evaluations y = log(f(x)) \item{aOpt, bOpt}{Analytically normalized
#'   parameters of f(x).} \item{logLike}{Log-likelihood of f(x)} \item{y}{Vector
#'   with values y_i = log(f(X_)) of the normalized density (\eqn{logLike =
#'   \sum(y_i)}).} \item{aOptSparse, bOptSparse}{Sparse parametrization
#'   normalized on the integration grid.}
#'
#' @example Examples/correctIntegral

fmlcd <- function(X, w=rep(1/nrow(X),nrow(X)), init='', verbose=0, intEps = 1e-4, objEps = 1e-7, offset = 1e-1) {

  gamma = 1000
  n <- dim(X)[1]
  d <- dim(X)[2]

  ## renormalize weights to sum to one
  w <- w/sum(w)

  ## substract the mean from X
  mu = apply(X,2,mean)
  X = X - mu

  ## obtain faces of convex hull of X
  cvhParams <- calcCvxHullFaces(X)

  paramsKernel <- double()
  if (init == 'kernel') {
    if (verbose > 0) {
      print("Use a kernel density estimation for initialization.")
    }
      params <- paramFitKernelDensity(X, w, cvhParams$cvh)
  } else if (init == 'smooth') {
      if (verbose > 0) {
        print("Use a smooth log-concave density estimation for initialization.")
      }
      params <- paramFitGammaOne(X, w, cvhParams$ACVH, cvhParams$bCVH, cvhParams$cvh)
  } else {
    if (n < 2500) {
      if (verbose > 0) {
        print("Choose between a kernel density and a smooth log-concave density estimation for initialization.")
      }
      paramsKernel <- paramFitKernelDensity(X, w, cvhParams$cvh)
      params <- paramFitGammaOne(X, w, cvhParams$ACVH, cvhParams$bCVH, cvhParams$cvh)
    } else {
      if (verbose > 0) {
        print("Use a smooth log-concave density estimation for initialization.")
      }
      params <- paramFitGammaOne(X, w, cvhParams$ACVH, cvhParams$bCVH, cvhParams$cvh)
    }
  }

  # The larger vector has to be params (due to C conventions)
  if (length(paramsKernel) > length(params)) {
    paramsTmp <- params
    params <- paramsKernel
    paramsKernel <- params
   }

  # call C code that optimizes the MLE objective with initial parameters params and paramsKernel
  # optimal set of parameters choosen inside the function
  res <- callNewtonBFGSLC(X, w, params, paramsKernel, cvhParams, gamma, verbose, intEps, objEps, offset)
  result <- correctIntegral(X, mu, res$a, res$b, cvhParams$cvh);
  logLike = result$logMLE * t(w) * length(w);

  # update convex hull parameters for true X
  #r$bCVH <- r$bCVH + r$ACVH %*% t(mu)

  return(c(result,list("logLike" = logLike)))
}
