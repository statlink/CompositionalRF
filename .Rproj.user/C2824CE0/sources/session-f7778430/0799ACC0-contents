comp.rf <- function(xnew = x, y, x, type = "alr", ntrees, nfeatures, minleaf) {

  if ( type == "alr" )  y <- Compositional::alr(y)
  est <- MultivariateRandomForest::build_forest_predict(trainX = x, trainY = y,
         n_tree = ntrees, m_feature = nfeatures, min_leaf = minleaf, testX = xnew)
  Compositional::alrinv(est)
}

