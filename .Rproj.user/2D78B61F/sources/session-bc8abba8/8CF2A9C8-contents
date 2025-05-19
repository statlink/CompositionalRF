comp.rf <- function(xnew = x, y, x, type = "alr", ntrees, nfeatures, minleaf,
                    ncores = 1) {
  
  # if ( type == "alr" )  y <- Compositional::alr(y)
  # est <- MultivariateRandomForest::build_forest_predict(trainX = x,
  #                                                      trainY = y,
  #   n_tree = ntrees, m_feature = nfeatures, min_leaf = minleaf, testX = xnew)
  if ( type == "alr" )  y <- alrOptimized(y)
  est <- mrf(xnew      = xnew,
             y         = y,
             x         = x,
             ntrees    = ntrees,
             nfeatures = nfeatures,
             minleaf   = minleaf,
             ncores    = ncores)
  Compositional::alrinv(est)
}
