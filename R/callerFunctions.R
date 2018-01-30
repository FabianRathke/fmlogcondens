callNewtonBFGSLC <- function (X, w, params, paramsKernel, cvhParams, gamma, verbose) {
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
              as.double(1e-4),
              as.double(1e-7),
              as.double(1e-1),
              as.integer(verbose),
              as.double(gamma))
  # res$lenP denotes the number of active parameters in res$params
  optParams <- r$params[1:r$lenP]

  nH <- r$lenP/(d+1) # number of hyperplanes
  a <- matrix(optParams[1:(d * nH)],nH,d)
  b <- optParams[(d*nH+1):length(optParams)]
  return(list("a" = a, "b" = b))
}

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

callCalcExactIntegralC <- function(X, y, P, Q, eps) {

  n <- dim(X)[1]
  d <- dim(X)[2]

  T <- geometry::convhulln(rbind(P, Q))
  T <- T[!(apply(T, 1, max) > n), ]

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

  return(r)
}
