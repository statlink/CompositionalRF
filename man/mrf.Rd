\name{mrf}
\alias{mrf}
\title{
Multivariate Random Forests
}

\description{
Multivariate Random Forests.
}

\usage{
mrf(xnew, y, x, ntrees, nfeatures, minleaf, ncores = 1)
}

\arguments{
\item{xnew}{
A matrix with the new predictor variables whose multivariate response values are
to be predicted.
}
\item{y}{
The response multivariate data. 
}
\item{x}{
A matrix with the predictor variables data.
}
\item{ntrees}{
The number of trees to construct in the random forest.
}
\item{nfeatures}{
The number of randomly selected predictor variables considered for a split in
each regression tree node,
which must be less than the number of input precictors.
}
\item{minleaf}{
Minimum number of observations in the leaf node. If a node has less than or
equal to minleaf observations,
there will be no splitting in that node and this node will be considered as a
leaf node.
The number evidently must be less than or equal to the sample size.
}
\item{ncores}{
The number of cores to use. If greater than 1, parallel computing will take
place. It is advisable to use it if you have many observations and or many
variables, otherwise it will slow down the process. The default is 1, meaning
that code is executed serially.
}
}

\details{
Multivariate random forest algorithm of Rahman, Otridge and Pal (2017) is
applied.
}

\value{
A matrix with the estimated multivariate response values.
}

\references{
Rahman R., Otridge J. and Pal R. (2017). IntegratedMRF: random forest-based
framework for integrating prediction from different data types. Bioinformatics,
33(9): 1407--1410.

Segal M. and Xiao Y. (2011). Multivariate random forests. Wiley
Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 1(1): 80--87.
}

\author{
Christos Adam.

R implementation and documentation: Christos Adam \email{pada4m4@gmail.com} and
Michail Tsagris \email{mtsagris@uoc.gr}.
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
x <- matrix( rnorm(150 * 10), ncol = 10 )
mod <- mrf(x[1:10, ], y, x, ntrees = 2, nfeatures = 5, minleaf = 10)
mod
}

