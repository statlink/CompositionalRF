\name{cv.comprf}
\alias{cv.comprf}
\title{
Cross-Validation of the Compositional Random Forests
}

\description{
Cross-Validation of the Compositional Random Forests.
}

\usage{
cv.comprf(y, x, ntrees = c(50, 100, 500, 1000), nfeatures, minleaf,
folds = NULL, nfolds = 10, seed = NULL, ncores = 1)
}

\arguments{
\item{y}{
The response compositional data. Zero values are not allowed.
}
\item{x}{
A matrix with the predictor variables data.
}
\item{ntrees}{
A vector with the possible number of trees to consider each time.
}
\item{nfeatures}{
A vector with the number of randomly selected predictor variables considered
for a split in each regression tree node.
}
\item{minleaf}{
A vector with the minimum number of observations in the leaf node.
}
\item{folds}{
If you have the list with the folds supply it here. You can also leave it
NULL and it will create folds.
}
\item{nfolds}{
The number of folds in the cross validation.
}
\item{seed}{
You can specify your own seed number here or leave it NULL.
}
\item{ncores}{
The number of cores to use. If greater than 1, parallel computing will take
place. It is advisable to use it if you have many observations and or many
variables, otherwise it will slow down the process.
}
}

\details{
K-fold cross-validation for the multivariate random forest with compositional
responses is performed.
}

\value{
A list including:
\item{kl}{
A matrix with the configurations of hyper-parameters tested and the estimated
Kullback-Leibler divergence, for each configuration.
}
\item{js}{
A matrix with the configurations of hyper-parameters tested and the estimated
Jensen-Shannon divergence, for each configuration.
}
}

\references{
Rahman R., Otridge J. and Pal R. (2017). IntegratedMRF: random forest-based
framework for integrating prediction from different data types. Bioinformatics,
33(9): 1407--1410.

Segal M. and Xiao Y. (2011). Multivariate random forests.
Wiley Interdisciplinary Reviews: Data mining and Knowledge Discovery, 1(1):
80--87.
}

\author{
Michail Tsagris.

R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr}.
}

%\note{
%%  ~~further notes~~
%}

\seealso{
\code{\link{comp.rf}
}
}

\examples{
y <- as.matrix(iris[, 1:4])
y <- y/ rowSums(y)
x <- matrix( rnorm(150 * 10), ncol = 10 )
mod <- cv.comprf(y, x, ntrees = 2, nfeatures = 5, minleaf = 10, nfolds = 2)
}

