alfa.comp.rf <- function(xnew = x, y, x, a = seq(-1, 1, by = 0.1), ntrees, nfeatures, minleaf) {

  config <- as.matrix( expand.grid(ntrees = ntrees, nfeatures = nfeatures, minleaf = minleaf) )
  p <- dim(config)[1]
  est <- list()
  if ( min(y) == 0 )  a <- a[a > 0]
  la <- length(a)
  if ( !is.matrix(xnew) )  xnew <- as.matrix(xnew)
  names <- paste("alpha", a)
  est <- sapply(names, function(x) NULL)

  for (i in 1:la) {
    ya <- Compositional::alfa(y, a[i])$aff
    for (j in 1:p) {
      yhat <- MultivariateRandomForest::build_forest_predict(trainX = x, trainY = ya,
              n_tree = config[j, 1], m_feature = config[j, 2], min_leaf = config[j, 3], testX = xnew)
      est[[ i ]][[ j ]] <- Compositional::alfainv(yhat, a[i])
    }
  }

  est
}

