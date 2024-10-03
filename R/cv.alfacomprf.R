cv.alfacomprf <- function(y, x, a = seq(-1, 1, by = 0.1), ntrees = c(50, 100, 200), nfeatures,
                          minleaf, folds = NULL, nfolds = 10, seed = NULL, ncores = 1) {

  if ( min(y) == 0 )  a <- a[a > 0]
  n <- dim(y)[1]

  if (ncores > 1) {
    cl <- parallel::makePSOCKcluster(ncores)
    doParallel::registerDoParallel(cl)
    if ( is.null(folds) )
      folds <- Compositional::makefolds(1:n, nfolds = nfolds, stratified = FALSE, seed = seed )
    nfolds <- length(folds)
    p <- dim(config)[1]
    kl <- js <- matrix(nrow = nfolds, ncol = p)

    per <- foreach( k = 1:nfolds, .combine = rbind, .export = "alfa.comp.rf",
      .packages = c("MultivariateRandomForest") ) %dopar% {
       ytrain <- ly[ -folds[[ k ]], ]
       ytest <- y[ folds[[ k ]],  ]
       xtrain <- x[-folds[[ k ]], ]
       xtest <- x[folds[[ k ]], ]
       est <- CompositionalRF::alfa.comp.rf(xtest, ytrain, xtrain, a = a,
                               ntrees = ntrees, nfeatures = nfeatures, minleaf = mninleaf)
       for (j in 1:p) {
         ela <- abs( ytest * log( ytest / est ) )
         ela[ is.infinite(ela) ] <- NA
         kl[k, j] <- 2 * mean(ela , na.rm = TRUE)
         M <- 0.5 * (ytest + est[[ i ]][[ j ]])
         ela2 <- ytest * log( ytest / M ) + est[[ i ]][[ j ]] * log( est[[ i ]][[ j ]] / M )
         ela2[ is.infinite(ela2) ] <- NA
         js[k, j] <- mean(ela2, na.rm = TRUE)
       }
       return( cbind(kl, js) )
    }
    parallel::stopCluster(cl)
    kl <- Rfast::colmeans( per[, 1:(0.5 * p)] )
    js <- Rfast::colmeans( per[, (0.5 * p + 1):p] )
    kl <- cbind(config, kl )
    js <- cbind(config, js )

  } else {
    kla <- jsa <- list()
    names <- paste("alpha", a)
    kla <- jsa <- sapply(names, function(x) NULL)

    if ( is.null(folds) )
      folds <- Compositional::makefolds(1:n, nfolds = nfolds, stratified = FALSE, seed = seed)
    nfolds <- length(folds)
    p <- dim(config)[1]
    kl <- js <- matrix(nrow = nfolds, ncol = p)

    for ( k in 1:nfolds ) {
      ytrain <- y[ -folds[[ k ]], ]
      ytest <- y[ folds[[ k ]],  ]
      xtrain <- x[-folds[[ k ]], ]
      xtest <- x[folds[[ k ]], ]
      est <- CompositionalRF::alfa.comp.rf(xtest, ytrain, xtrain, a = a,
                                ntrees = ntrees, nfeatures = nfeatures, minleaf = minleaf)
      for (i in 1:la) {
        for (j in 1:p) {
          ela <- abs( ytest * log( ytest / est[[ i ]][[ j ]] ) )
          ela[ is.infinite(ela) ] <- NA
          kl[k, j] <- 2 * mean(ela , na.rm = TRUE)
          M <- 0.5 * (ytest + est[[ i ]][[ j ]])
          ela2 <- ytest * log( ytest / M ) + est[[ i ]][[ j ]] * log( est[[ i ]][[ j ]] / M )
          ela2[ is.infinite(ela2) ] <- NA
          js[k, j] <- mean(ela2, na.rm = TRUE)
        }
        kla[[ i ]] <- kl
        jsa[[ i ]] <- js
      }
    }  ##  end  for ( k in 1:nfolds ) {

    kla2 <- matrix(nrow = la, ncol = p)
    rownames(kla2) <- paste("alpha=", a, sep = "")
    jsa2 <- kla2
    for (i in 1:la) {
      kla2[i, ] <- Rfast::colmeans(kla[[ i ]])
      jsa2[i, ] <- Rfast::colmeans(jsa[[ i ]])
    }
  }

  list(config = config, kl = kla2, js = jsa2)
}







