#' @title Parameter Initialization Based on a Kernel Density
#'
#' @description \code{paramFitKernelDensity} first fits a kernel density to a
#'   sample X with weight vector w. It then calculates the parameters of the
#'   piecewise linear function defined to be the upper convex hull of
#'   (X,log(y)).
#'
#' @param X Set of data points (one sample per row)
#' @param w Vector with weights for X (\code{sum(w) == 1})
#' @param cvh Matrix where each row is a set of indices of points in X
#'   describing one face of conv(X)
#' @param h Scalar parameter that governs the Gaussian kernel
#'
#' @return A list containing the description of the upper convex hull of
#'   (X,log(y)) in term of hyperplane parameters: \item{a}{A matrix where each
#'   row constitutes the normal vector of a face} \item{b}{A vector where each
#'   entry constitutes the offset of a face}
#'
#' @example Examples/paramFitKernelDensity

paramFitKernelDensity <- function(X, w, cvh, h = apply(X, 2, sd) * n ** (-1 / (d+4))) {
  n <- dim(X)[1]
  d <- dim(X)[2]

  # C-function that calculates the kernel density
  r <- .C("calcKernelDens", X = t(X), sampleWeights = w, yT = matrix(0, n), h = h, n = n, d = d)
  yT = log(r$yT)

  # find upper convex hull of X and y
  finiteVals <- is.finite(yT)
  idxCVH <- setdiff(unique(as.vector(cvh)), which(!finiteVals))
  P <- matrix(c(X[finiteVals, ], yT[finiteVals]), nrow = length(finiteVals))
  Q <- matrix(c(X[idxCVH, ], rep(min(yT[idxCVH]) - 1, length(idxCVH))), nrow = length(idxCVH))
  T <- geometry::convhulln(rbind(P, Q))
  T <- T[!(apply(T, 1, max) > length(finiteVals)), ]

  # calls a C function that modifies yT such that the resulting log-concave density defined by X and y
  # normalizes to one
  r <- .C(  "calcExactIntegralC",
          as.double(t(X[finiteVals, ])),
          as.double(yT[finiteVals]),
          as.integer(t(T-1)),
          as.integer(dim(T)[1]),
          as.integer(length(finiteVals)),
          as.integer(d),
          as.double(1),
          as.double(0.01),
          a = as.double(matrix(0, nrow(T) * d)),
          b = as.double(matrix(0, nrow(T))))

  return(c(t(matrix(r$a, d, length(r$a) / d)), r$b))
  }
