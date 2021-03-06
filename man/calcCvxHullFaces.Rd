% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/calcCvxHullFaces.R
\name{calcCvxHullFaces}
\alias{calcCvxHullFaces}
\title{Calculates the convex hull of X}
\usage{
calcCvxHullFaces(X)
}
\arguments{
\item{X}{Set of data points (one sample per row)}
}
\value{
A list consisting of parameters describing the convex hull of X:
  \item{cvh}{A matrix of indices where each row constitutes one face}
  \item{ACVH}{A matrix where each row constitutes the normal vector of a
  face} \item{bCVH}{A vector where each entry constitutes the offset of the
  hyperplane for a face}
}
\description{
\code{calcCvxHullFaces} returns the parameters of the convex
  hull of X: The indices of points in X that compose the faces of conv(X)
  (one row per face) as well as the parameters of the hyperplanes that
  describe these faces.
}
