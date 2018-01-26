#' @title Estimates a Log-Concave Density
#'
#' @description \code{fastMLELogDens} returns a MLE estimate of a log-concave
#'   density for \code{X}. After obtaining an initial parameter estimate the MLE
#'   objective with log-concavity and normalization constraint is optimized
#'   using a quasi-Newton approach for large scale optimization (BFGS-L). The
#'   logarithm of the optimal density f(x) is a piecewise-linear function. Its
#'   parametrization in terms of a set of hyperplanes is returned.
#'
#' @param X Set of data points (one sample per row)
#' @param w Sample weights
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
fastMLELogDens <- function(X,
                      w=rep(1/nrow(X),nrow(X)),
                      init='',
                      verbose=1) {

  gamma = 1000

  dyn.load("~/Documents/Arbeit/Code/MyCode/LogConcave/R/kernelDensC.so")
  dyn.load("~/Documents/Arbeit/Code/MyCode/LogConcave/R/calcExactIntegral.so")
  dyn.load("~/Documents/Arbeit/Code/MyCode/LogConcave/R/bfgsInitC.so")
  dyn.load("~/Documents/Arbeit/Code/MyCode/LogConcave/R/bfgsFullC.so")

  n <- dim(X)[1]
  d <- dim(X)[2]

  ## renormalize weights to sum to one
  w <- w/sum(w)

  ## substract the mean from X
  mu = apply(X,2,mean)
  X = X - mu

  ## obtain faces of convex hull
  r <- calcCvxHullFaces(X)

  paramsKernel <- double()
  if (init == 'kernel') {
      params <- paramFitKernelDensity(X,w,r$cvh)
  } else if (init == 'smooth') {
      params <- paramFitGammaOne(X,w,r$ACVH,r$bCVH,r$cvh)
  } else {
    if (n < 2500) {
      paramsKernel <- paramFitKernelDensity(X,w,r$cvh)
      params <- paramFitGammaOne(X,w,r$ACVH,r$bCVH,r$cvh)
    } else {
      params <- paramFitGammaOne(X,w,r$ACVH,r$bCVH,r$cvh)
    }
  }

   if (length(paramsKernel) > length(params)) {
    paramsTmp <- params
    params <- paramsKernel
    paramsKernel <- params
  }

  res <- .C(  "newtonBFGSLC",
            as.double(X),
            as.double(w),
            as.double(c(apply(X,2,min),apply(X,2,max))),
            params = as.double(params),
            as.double(paramsKernel),
            lenP = as.integer(length(params)),
            as.integer(length(paramsKernel)),
            as.integer(d),
            as.integer(n),
            as.double(r$ACVH),
            as.double(r$bCVH),
            as.integer(length(r$bCVH)),
            as.double(1e-3),
            as.double(1e-7),
            as.double(1e-2),
            as.integer(2))
  # res$lenP denotes the number of active parameters in res$params
  optParams <- res$params[1:res$lenP]
  nH <- res$lenP/(d+1) # number of hyperplanes
  aOpt <- matrix(optParams[1:(d*nH)],nH,d)
  bOpt <- optParams[(d*nH+1):length(optParams)]

  result <- correctIntegral(X,mu,aOpt,bOpt,r$cvh);
  logLike = result$yT*t(w)*length(w);

  # update convex hull parameters for true X
  #r$bCVH <- r$bCVH + r$ACVH %*% t(mu)

  return(c(result,list("logLike" = logLike)))
}
