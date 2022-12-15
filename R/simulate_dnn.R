

#' @export
simulate.layer_dnn <- function(obj, data, balance.correction, balance.var) {

  select <- obj$filter(data)

  f <- as.formula(obj$formula)
  label <- as.character(terms(f)[[2]])

  data_baked <- bake(obj$data_recipe, new_data = data[select,])
  if(ncol(data_baked) == 1)
    data_baked <- data_baked %>% mutate(intercept = 1)

  x <- select(data_baked,-as.name(label)) %>% as.matrix()

  def_x <- def_x_mlp(obj$method_options$use_embedding,
                     obj$method_options$embedding_var,
                     f,
                     data[select,],
                     data_baked,
                     obj$data_recipe,
                     label)

  if(!obj$method_options$use_embedding) {
    x.inputs <- list(def_x$x)
  }
  else {
    x.inputs <- list(def_x$x_no_fact,def_x$x_fact)
    x.inputs[sapply(x.inputs, is.null)] <- NULL
  }

  if(!is.null(obj$method_options$ae.hidden)) {
    x <- data.frame(obj$model_en %>% predict(x)) %>% as.matrix()
  }

  if(!obj$method_options$bias_regularization) {
    response <- predict(obj$fit, x.inputs)
  }
  else {
    Zlearn   <- data.frame(obj$zz %>% predict(x.inputs))
    names(Zlearn) <- paste0('X', 1:ncol(Zlearn))
    response <- predict(obj$fit, newdata = Zlearn, type = 'response') %>% as.matrix()
  }

  if(balance.correction) {
    response <- response * obj$balance.correction[as.character((data[select,])[[balance.var]])]
  }

  if(obj$method_options$distribution == 'bernoulli') {
    simulation <- runif(dim(response)[1]) < response
  } else if(obj$method_options$distribution == 'gaussian') {
    simulation <- rnorm(dim(response)[1], mean = as.vector(response), sd = obj$sigma)
  } else if(obj$method_options$distribution == 'gamma') {
    simulation <- rgamma(dim(response)[1], scale = as.vector(response) / obj$shape, shape = obj$shape)
  }

  if(!is.null(obj$transformation)) {
    simulation <- obj$transformation$inverse_transform(simulation)
  }

  if(obj$method_options$step_log) simulation <- exp(simulation)

  ret <- rep(0, nrow(data))
  ret[select] <- simulation

  return(ret)
}

