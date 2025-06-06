alfa.comp.rf <- function(xnew = x, y, x, a = seq(-1, 1, by = 0.1), ntrees, nfeatures, 
                         minleaf, ncores = 1) {

  config <- as.matrix( expand.grid(ntrees = ntrees, nfeatures = nfeatures, minleaf = minleaf) )
  p <- dim(config)[1]
  est <- list()
  if ( min(y) == 0 )  a <- a[a > 0]
  la <- length(a)
  if ( !is.matrix(xnew) )  xnew <- as.matrix(xnew)
  names <- paste("alpha", a)
  est <- sapply(names, function(x) NULL)

  for ( i in 1:la ) {
    ya <- Compositional::alfa(y, a[i])$aff
    for (j in 1:p) {
      yhat <- CompositionalRF::mrf(xnew = xnew, y = ya, x = x, ntrees = config[j, 1], 
                                   nfeatures = config[j, 2], minleaf = config[j, 3], ncores = ncores)
      est[[ i ]][[ j ]] <- Compositional::alfainv(yhat, a[i])
    }
  }
  
  est
}

