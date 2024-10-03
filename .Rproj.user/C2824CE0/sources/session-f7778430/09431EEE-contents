alfa.comp.rf <- function(xnew = x, y, x, a = seq(-1, 1, by = 0.1), ntrees, nfeatures, minleaf) {

  est <- list()
  if ( min(y) == 0 )  a <- a[a > 0]
  la <- length(a)
  if ( !is.matrix(xnew) )  xnew <- as.matrix(xnew)
  names <- paste("alpha", a)
  est <- sapply(names, function(x) NULL)

  for (i in 1:la) {
    ya <- Compositional::alfa(y, a[i])$aff
    yhat <- MultivariateRandomForest::build_forest_predict(trainX = x, trainY = ya,
                           n_tree = ntrees, m_feature = nfeatures, min_leaf = minleaf, testX = xnew)
    est[[ i ]] <- Compositional::alfainv(yhat, a[i])
  }
  est

}

