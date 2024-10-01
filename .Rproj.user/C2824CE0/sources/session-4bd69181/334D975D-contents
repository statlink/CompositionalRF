\name{comp.rf}
\alias{comp.rf}
\title{
Compositional Random Forests
}

\description{
Compositional Random Forests.
}

\usage{
comp.rf(xnew = x, y, x, type = "alr", ntrees, nfeatures, minleaf)
}

\arguments{
\item{xnew}{
A matrix with the new predictor variables whose compositional response values are to be predicted.
}
\item{y}{
The response compositional data. Zero values are not allowed.
}
\item{x}{
A matrix with the predictor variables data.
}
\item{type}{
If the responses are alreay transformed with the additive log-ratio transformation type 0, otherwise,
if they are compositional data, leave it equal to "alr", so that the data will be transformed.
}
\item{ntrees}{
The number of trees to construct in the random forest.
}
\item{nfeatures}{
The number of randomly selected predictor variables considered for a split in each regression tree node,
which must be less than the number of input precictors.
}
\item{minleaf}{
Minimum number of observations in the leaf node. If a node has less than or equal to minleaf observations,
there will be no splitting in that node and this node will be considered as a leaf node.
The number evidently must be less than or equal to the sample size.
}
}

\details{
The compositional are first log-transformed using the additive log-ratio transformation and then the multivariate random forest algorithm of Rahman, Otridge and Pal (2017) is applied.
}

\value{
A matrix with the estimated compositional response values.
}

\references{
Rahman R., Otridge J. and Pal R. (2017). IntegratedMRF: random forest-based framework for
integrating prediction from different data types. Bioinformatics, 33(9): 1407--1410.

Segal M. and Xiao Y. (2011). Multivariate random forests.
Wiley Interdisciplinary Reviews: Data mining and Knowledge Discovery, 1(1): 80--87.
}

\author{
Michail Tsagris.

R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr}.
}

%\note{
%%  ~~further notes~~
%}

\seealso{
\code{\link{cv.comprf}
}
}

\examples{
y <- as.matrix(iris[, 1:4])
y <- y/ rowSums(y)
x <- matrix( rnorm(150 * 10), ncol = 10 )
mod <- comp.rf(x[1:10, ], y, x, ntrees = 2, nfeatures = 5, minleaf = 10)
mod
}

