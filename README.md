# fmlogcondens

`fmlogcondens` is a package written in R and C, which provides a solver for the maximum likelihood estimation of a multivariate log-concave density. Log-concave densities are an important class of density functions, which encompass several well-known parametric densities such as the normal, uniform, gamma(r, lambda) with r >= 1, Beta(a,b) with a,b >= 1 and many more. They are density functions, whose logarithm is a concave function.  

Estimating a log-concave density for a sample X is straight-forward using our package:

```R
install.packages("fmlogcondens")
library(fmlogcondens)

# sample data points
X <- matrix(rnorm(200),100,2)
# get the MLE for a log-concave density
params <- fmlcd(X)
```

For the ML estimate, the optimal solution in log-space is a piecewise linear concave function. Our estimator thus returns a set of hyperplane parameters stored in `params$a` and `params$b`. 

For plotting, we use the facilities of the package `LogConcDEAD`

```R
install.packages("LogConcDEAD")
library(LogConcDEAD)

# convert our params to an object of type LogConcDEAD
r <- LogConcDEAD::getinfolcd(X, params$logMLE)
# plot the estimated density
par(mfrow = c(1, 2)) #square plots
plot(r, addp = FALSE, asp = 1)
plot(r, uselog = TRUE, addp = FALSE, asp = 1)
```

For more details, please see the [documentation](vignettes/documentation.html) and check the help `?fmlcd`.

If you found an error or have a questions regarding the package, feel free to contact me: Fabian Rathke (frathke at gmail dot com).