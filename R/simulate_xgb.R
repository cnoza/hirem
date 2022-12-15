
#' @export
simulate.layer_xgb <- function(obj, data, balance.correction, balance.var) {

  select <- obj$filter(data)
  #f <- as.formula(paste0(obj$formula,'-1'))
  f <- as.formula(obj$formula)
  label <- as.character(terms(f)[[2]])

  data_baked <- bake(obj$data_recipe, new_data = data[select,])
  newdata <- select(data_baked,-as.name(label)) %>% as.matrix()

  ret <- rep(0, nrow(data))

  if(!is.logical(newdata)) {

    newdata.xgb <- xgb.DMatrix(data = newdata, info = list('label' = as.matrix(data[select,label])))
    response <- predict(obj$fit, ntreelimit = obj$fit$niter, newdata = newdata.xgb, type = 'response')

    if(balance.correction) {
      response <- response * obj$balance.correction[as.character((data[select,])[[balance.var]])]
    }

    if(obj$method_options$objective == 'binary:logistic') {
      simulation <- runif(length(response)) < response
    } else if(obj$method_options$objective == 'reg:squarederror') {
      simulation <- rnorm(length(response), mean = response, sd = obj$sigma)
    } else if(obj$method_options$objective == 'reg:gamma') {
      simulation <- rgamma(length(response), scale = response / obj$shape, shape = obj$shape)
    }

    if(!is.null(obj$transformation)) {
      simulation <- obj$transformation$inverse_transform(simulation)
    }

    ret[select] <- simulation

  }

  #ret <- rep(0, nrow(data))
  #ret[select] <- simulation

  return(ret)
}
