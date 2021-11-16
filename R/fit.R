#' @export
fit <- function(object, ...) {
  UseMethod("fit")
}

fit.layer_glm <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat("Fitting layer_glm ...\n")

  layer$formula <- formula

  data <- obj$data_training

  if(!training) {
    data <- obj$data_observed
  }

  if(!is.null(fold)) {
    data <- data %>% filter(cv_fold != fold)
  }

  data.filter <- data[layer$filter(data), ]

  if(!is.null(layer$transformation)) {
    label <- as.character(terms(as.formula(formula))[[2]])
    data.filter[,label] <- layer$transformation$transform(data.filter[,label])
  }

  if(!is.null(obj$weights))
    weights.vec <- obj$weights[data.filter[[obj$weight.var]]] else
      weights.vec <- NULL

  if(layer$method_options$family == 'Gamma') {
    layer$fit <- glm(as.formula(formula), data = data.filter, family = Gamma(link = log),
                     weights = weights.vec)

  } else {
    layer$fit <- glm(as.formula(formula), data = data.filter, family = layer$method_options,
                     weights = weights.vec)
  }

  if(!is.null(obj$balance.var)){
    layer$balance.correction <- sapply(data.filter %>% split(data.filter[[obj$balance.var]]),
                                     function(x) sum(x[[layer$name]])/sum(predict(layer$fit, newdata = x, type = 'response')))
  }

  if(layer$method_options$family == 'gaussian') {
    layer$sigma <- sd(layer$fit$residuals)
  } else if(layer$method_options$family == 'Gamma') {
    shape <- hirem_gamma_shape(observed = layer$fit$y, fitted = layer$fit$fitted.values, weight = layer$fit$weights)
    layer$shape <- shape$shape
    layer$shape.se <- shape$se
  }

  return(layer)
}

#' @export
fit.layer_gbm <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat("Fitting layer_gbm ...\n")
  layer$formula <- formula

  data <- obj$data_training
  if(!training) {
    data <- obj$data_observed
  }

  if(!is.null(fold)) {
    data <- data %>% filter(cv_fold != fold)
  }

  data <- data[layer$filter(data), ]

  if(!is.null(layer$transformation)) {
    label <- as.character(terms(as.formula(formula))[[2]])
    data[,label] <- layer$transformation$transform(data[,label])
  }

  weights.vec <- if(is.null(obj$weights)) NULL else obj$weights[data[[obj$weight.var]]]

  layer$fit <- gbm(as.formula(formula),
                   data = data,
                   distribution = layer$method_options$distribution,
                   n.trees = layer$method_options$n.trees,
                   cv.folds = layer$method_options$cv,
                   interaction.depth = layer$method_options$interaction.depth,
                   shrinkage = layer$method_options$shrinkage,
                   n.minobsinnode = layer$method_options$n.minobsinnode,
                   bag.fraction = layer$method_options$bag.fraction,
                   weights = weights.vec,
                   keep.data = TRUE)

  if(layer$method_options$select_trees == 'last') {
    layer$iter <- layer$method_options$n.trees
  } else {
    layer$iter <- gbm.perf(layer$fit, plot.it = FALSE)

    if(length(layer$iter) == 0) {
      layer$iter <- which.min(layer$fit$oobag.improve[is.finite(layer$fit$oobag.improve)])
    }
  }

  if(!is.null(obj$balance.var)){
    layer$balance.correction <- sapply(data %>% split(data[[obj$balance.var]]),
                                       function(x) sum(x[[layer$name]])/sum(predict(layer$fit, newdata = x, n.trees = layer$iter, type = 'response')))
  }

  if(layer$method_options$distribution == 'gaussian') {
    layer$sigma <- sd(predict(layer$fit, n.trees = layer$iter, type = "response") - layer$fit$data$y)
  }

  if(layer$method_options$distribution == 'gamma') {
    shape <- hirem_gamma_shape(observed = layer$fit$data$y, fitted = predict(layer$fit, n.trees = layer$iter, type = "response"),
                               weight = layer$fit$data$w)
    layer$shape <- shape$shape
    layer$shape.se <- shape$se
  }

  return(layer)
}

#' @export
fit.layer_xgb <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat("Fitting layer_xgb ...\n")
  layer$formula <- formula

  data <- obj$data_training
  if(!training) {
    data <- obj$data_observed
  }

  if(!is.null(fold)) {
    data <- data %>% filter(cv_fold != fold)
  }

  data <- data[layer$filter(data), ]
  f <- as.formula(formula)
  label <- as.character(terms(f)[[2]])

  if(!is.null(layer$transformation)) {
    data[,label] <- layer$transformation$transform(data[,label])
  }

  data.xgb <- xgb.DMatrix(data = as.matrix(sparse.model.matrix(f, data=data)[,-1]),
                          info = list(
                            'label' = as.matrix(data[,label])
                          ))

  if(!is.null(layer$method_options$nfolds)) {
    if(!is.null(layer$method_options$hyper_grid)) {
      hyper_grid <- layer$method_options$hyper_grid
    }
    else {
      hyper_grid <- expand.grid(
        eta = 0.01,
        max_depth = c(3,6),
        min_child_weight = 1000,
        subsample = c(0.5,0.75),
        colsample_bytree = 0.5,
        gamma = c(0),
        lambda = c(0,.1),
        alpha = c(0,.1)
      )
    }
    best_eval_metric = Inf

    cat('... cross-validation started ... \n')

    for(i in seq_len(nrow(hyper_grid))) {

      params = list(
        booster = "gbtree",
        eval_metric = layer$method_options$eval_metric,
        eta = hyper_grid$eta[i],
        max_depth = hyper_grid$max_depth[i],
        min_child_weight = hyper_grid$min_child_weight[i],
        subsample = hyper_grid$subsample[i],
        colsample_bytree = hyper_grid$colsample_bytree[i],
        gamma = hyper_grid$gamma[i],
        lambda = hyper_grid$lambda[i],
        alpha = hyper_grid$alpha[i]
      )

      xval <- xgb.cv(
        data = data.xgb,
        nrounds = layer$method_options$nrounds,
        objective = layer$method_options$objective,
        early_stopping_rounds = layer$method_options$early_stopping_rounds,
        nfold = layer$method_options$nfolds,
        stratified = T,
        verbose = layer$method_options$verbose,
        params = params)


      min_eval_metric <- min(xval$evaluation_log[,4])
      min_eval_index <- as.numeric(which.min(as.matrix(xval$evaluation_log[,4])))

      if (min_eval_metric < best_eval_metric) {
        best_eval_metric = min_eval_metric
        best_eval_index = min_eval_index
        best_param = params
      }
    }

    nrounds <- best_eval_index
    params <- best_param

  }
  else {

    nrounds = layer$method_options$nrounds
    params = list(
      booster = layer$method_options$booster,
      objective = layer$method_options$objective,
      eval_metric = layer$method_options$eval_metric,
      eta = layer$method_options$eta,
      nthread = layer$method_options$nthread,
      subsample = layer$method_options$subsample,
      colsample_bynode = layer$method_options$colsample_bynode,
      max_depth = layer$method_options$max_depth,
      min_child_weight = layer$method_options$min_child_weight,
      gamma = layer$method_options$gamma,
      lambda = layer$method_options$lambda,
      alpha = layer$method_options$alpha
    )

  }

  layer$fit <- xgb.train(
    data = data.xgb,
    nrounds = nrounds,
    verbose = layer$method_options$verbose,
    params = params
  )

  layer$best_params <- params
  layer$best_iteration <- layer$fit$best_iteration
  layer$best_ntreelimit <- layer$fit$best_ntreelimit
  layer$best_score <- layer$fit$best_score
  layer$niter <- layer$fit$niter

  if(layer$method_options$objective == 'reg:squarederror') {
    layer$sigma <- sd(predict(layer$fit, newdata = data.xgb, type = "response") - as.matrix(data[,label]))
  }

  if(layer$method_options$objective == 'reg:gamma') {
    shape <- hirem_gamma_shape(as.matrix(data[,label]), predict(layer$fit, ntreelimit = obj$best_ntreelimit, newdata = data.xgb, type = "response"))
    layer$shape <- shape$shape
    layer$shape.se <- shape$se
  }

  return(layer)
}

#' @export
fit.layer_mlp_h2o <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat("Fitting layer_mlp_h2o ...\n")
  layer$formula <- formula

  data <- obj$data_training
  if(!training) {
    data <- obj$data_observed
  }

  if(!is.null(fold)) {
    data <- data %>% filter(cv_fold != fold)
  }

  data <- data[layer$filter(data), ]

  f <- as.formula(formula)
  label <- as.character(terms(f)[[2]])

  if(!is.null(layer$transformation)) {
    data[,label] <- layer$transformation$transform(data[,label])
  }

  h2o.init()
  h2o.no_progress()

  data.h2o <- as.h2o(data)
  layer$fit <- h2o.deeplearning(x = attr(terms(f),"term.labels"),
                                y = label,
                                training_frame = data.h2o,
                                distribution = layer$method_options$distribution,
                                hidden = layer$method_options$hidden,
                                nfolds = layer$method_options$nfolds,
                                epochs = layer$method_options$epochs,
                                train_samples_per_iteration = layer$method_options$train_samples_per_iteration,
                                reproducible = layer$method_options$reproducible,
                                activation = layer$method_options$activation,
                                single_node_mode = layer$method_options$single_node_mode,
                                balance_classes = layer$method_options$balance_classes,
                                force_load_balance = layer$method_options$force_load_balance,
                                seed = layer$method_options$seed,
                                tweedie_power = layer$method_options$tweedie_power,
                                score_training_samples = layer$method_options$score_training_samples,
                                score_validation_samples = layer$method_options$score_validation_samples,
                                stopping_rounds = layer$method_options$stopping_rounds,
                                input_dropout_ratio = layer$method_options$input_dropout_ratio,
                                hidden_dropout_ratios = layer$method_options$hidden_dropout_ratios)

  if(layer$method_options$distribution == 'gaussian') {
    layer$sigma <- sd(h2o.predict(layer$fit, data.h2o) - data[,label])
  }

  if(layer$method_options$distribution == 'gamma') {
    shape <- hirem_gamma_shape(data[,label], h2o.predict(layer$fit, data.h2o))
    layer$shape <- shape$shape
    layer$shape.se <- shape$se
  }


  return(layer)
}

#' @export
fit.layer_mlp_keras <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat("Fitting layer_mlp_keras ...\n")
  layer$formula <- formula

  data <- obj$data_training
  if(!training) {
    data <- obj$data_observed
  }

  if(!is.null(fold)) {
    data <- data %>% filter(cv_fold != fold)
  }

  data <- data[layer$filter(data), ]

  f <- as.formula(formula)
  label <- as.character(terms(f)[[2]])

  if(!is.null(layer$transformation)) {
    data[,label] <- layer$transformation$transform(data[,label])
  }

  data_recipe <- recipe(f, data=data)

  if(layer$method_options$step_log)
    data_recipe <- data_recipe %>% step_log(as.name(label))
  if(layer$method_options$step_normalize)
    data_recipe <- data_recipe %>% step_normalize(all_numeric(), -all_outcomes())

  data_recipe <- data_recipe %>% step_dummy(all_nominal(), one_hot = FALSE)
  data_recipe <- data_recipe %>% prep()
  layer$data_recipe <- data_recipe

  data_baked <- bake(data_recipe, new_data = data)
  if(ncol(data_baked) == 1)
    data_baked <- data_baked %>% mutate(intercept = 1)

  x <- select(data_baked,-as.name(label)) %>% as.matrix()
  y <- data_baked %>% pull(as.name(label))

  layer$x <- x
  layer$y <- y

  inputs <- layer_input(shape = c(ncol(x)), name='ae_input_layer')

  if(!is.null(layer$method_options$ae.hidden)) {

    nn <- length(layer$method_options$ae.hidden)

    ae_hidden_l      <- list()
    ae_hidden_l[[1]] <- layer_dense(inputs,
                                  units = layer$method_options$ae.hidden[1],
                                  activation = layer$method_options$ae.activation.hidden[1],
                                  name='ae_encoding_layer_1')
    if(nn>1) {
      for(i in 2:nn) {
        ae_hidden_l[[i]] <- layer_dense(ae_hidden_l[[i-1]],
                                        units = layer$method_options$ae.hidden[i],
                                        activation = layer$method_options$ae.activation.hidden[i],
                                        name = ifelse(i==nn,'ae_bottleneck_layer',paste0('ae_encoding_layer_',i)))
      }
      for(i in 1:(nn-1)) {
        ae_hidden_l[[nn+i]] <- layer_dense(ae_hidden_l[[nn+i-1]],
                                          units = layer$method_options$ae.hidden[nn-i],
                                          activation = layer$method_options$ae.activation.hidden[nn-i],
                                          name = paste0('ae_decoding_layer_',i))
      }
    }

    ae_output_l <- layer_dense(ae_hidden_l[[(2*nn-1)]], ncol(x), name='ae_output_layer')

    autoencoder <- keras_model(inputs, ae_output_l)
    model_en <- keras_model(inputs, ae_hidden_l[[nn]])

    # Autoencoder model
    summary(autoencoder)
    # Encoder model
    summary(model_en)

    autoencoder %>% compile(loss = 'mae', optimizer='adam')

    autoencoder %>% keras::fit(
      x = x,
      y = x,
      epochs = layer$method_options$epochs,
      batch_size = layer$method_options$batch_size
    )

    x <- model_en %>% predict(x)
    inputs <- layer_input(shape = c(ncol(x)))
    layer$x.encoded <- x
    layer$model_en <- model_en

  }

  if(layer$method_options$batch_normalization)
    output <- inputs %>% layer_batch_normalization()


  if(!is.null(layer$method_options$hidden)) {

    n <- length(layer$method_options$hidden)

    for(i in seq(from = 1, to=n)) {
      if(i==1 & !layer$method_options$batch_normalization) {
        output <- inputs %>% layer_dense(units = layer$method_options$hidden[i],
                                         name = ifelse(i==n,'last_hidden_layer',paste0('hidden_layer_',i))) %>%
          layer_activation(activation = layer$method_options$activation.hidden[i])
        if(!is.null(layer$method_options$dropout.hidden))
          output <- output %>% layer_dropout(rate = layer$method_options$dropout.hidden[i])
      }
      else {
        output <- output %>% layer_dense(units = layer$method_options$hidden[i],
                                         name = ifelse(i==n,'last_hidden_layer',paste0('hidden_layer_',i))) %>%
          layer_activation(activation = layer$method_options$activation.hidden[i])
        if(!is.null(layer$method_options$dropout.hidden))
          output <- output %>% layer_dropout(rate = layer$method_options$dropout.hidden[i])
      }
    }

  }

  # Initialization with the homogeneous model
  # See Ferrario, Andrea and Noll, Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April 23, 2020).
  # Available at SSRN: https://ssrn.com/abstract=3226852 or http://dx.doi.org/10.2139/ssrn.3226852 p.29.
  if(!is.null(layer$method_options$family_for_init)) {
    f.hom <- paste0(label, '~ 1')
    glm.hom <- glm(as.formula(f.hom),data = data, family = layer$method_options$family_for_init)
    layer$glm.hom <- glm.hom
    if(!layer$method_options$batch_normalization & is.null(layer$method_options$hidden))
      output <- inputs %>% layer_dense(units = 1, activation = layer$method_options$activation.output,
                                       weights = list(array(0,dim=c(ifelse(!is.null(layer$method_options$hidden),layer$method_options$hidden[n],c(ncol(x))),1)),
                                                      array(glm.hom$coefficients[1], dim=c(1))),
                                       use_bias = layer$method_options$use_bias,
                                       name = 'output_layer')
    else
      output <- output %>% layer_dense(units = 1, activation = layer$method_options$activation.output,
                                       weights = list(array(0,dim=c(ifelse(!is.null(layer$method_options$hidden),layer$method_options$hidden[n],c(ncol(x))),1)),
                                                      array(glm.hom$coefficients[1], dim=c(1))),
                                       use_bias = layer$method_options$use_bias,
                                       name = 'output_layer')
  }
  else {
    if(!layer$method_options$batch_normalization & is.null(layer$method_options$hidden))
      output <- inputs %>% layer_dense(units = 1, activation = layer$method_options$activation.output,
                                       use_bias = layer$method_options$use_bias,
                                       name = 'output_layer')
    else
      output <- output %>% layer_dense(units = 1, activation = layer$method_options$activation.output,
                                       name = 'output_layer')
  }

  model <- keras_model(inputs = inputs, outputs = c(output))
  print(summary(model))

  model %>% compile(
    loss = layer$method_options$loss,
    optimizer = layer$method_options$optimizer,
    metrics = layer$method_options$metrics
  )

  earlystopping <- callback_early_stopping(
    monitor = layer$method_options$monitor,
    patience = layer$method_options$patience)

  layer$history <- model %>%
    keras::fit(x, y, epochs = layer$method_options$epochs,
               batch_size = layer$method_options$batch_size,
               validation_split = layer$method_options$validation_split,
               callbacks = list(earlystopping))

  if(!layer$method_options$bias_regularization) {
    layer$fit <- model
  }
  else {
    # We keep track of the neural network (biased) model
    layer$fit.biased <- model
    # Source: Ferrario, Andrea and Noll, Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April 23, 2020), p.52
    glm.formula <- function(nb) {
      string <- "yy ~ X1 "
      if(nb>1) {for (i in 2:nb) {string <- paste(string,"+ X",i, sep="")}}
      string
    }

    zz        <- keras_model(inputs = model$input, outputs=get_layer(model,'last_hidden_layer')$output)
    layer$zz  <- zz

    Zlearn    <- data.frame(zz %>% predict(x))
    names(Zlearn) <- paste0('X', 1:ncol(Zlearn))

    # We keep track of the pre-processed data for analysis purposes
    layer$Zlearn <- Zlearn

    Zlearn$yy <- y

    if(layer$method_options$distribution == 'gamma')
      fam <- Gamma(link=log) # default link=inverse but we use exponential as activation function
    else if(layer$method_options$distribution == 'bernoulli')
      fam <- binomial() # default link = logit <-> activation function = sigmoid
    else if(layer$method_options$distribution == 'poisson')
      fam <- poisson() # default link = log <-> activation function = exponential
    else if(layer$method_options$distribution == 'gaussian')
      fam <- gaussian() # default link = identity <-> activation function = identity
    else
      stop('Bias regularization is not supported for this distribution.')

    glm1 <- glm(as.formula(glm.formula(ncol(Zlearn)-1)), data=Zlearn, family=fam)
    layer$fit <- glm1
  }

  if(layer$method_options$bias_regularization)
    pred <- layer$fit$fitted.values
  else
    pred <- layer$fit %>% predict(x)

  if(layer$method_options$distribution == 'gaussian') {
    layer$sigma <- sd(pred - y)
  }

  if(layer$method_options$distribution == 'gamma') {
    shape <- hirem_gamma_shape(observed = y, fitted = pred)
    layer$shape <- shape$shape
    layer$shape.se <- shape$se
  }

  return(layer)
}

#' @export
fit.layer_cann <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat("Fitting layer_cann ...\n")
  layer$formula <- formula

  data <- obj$data_training
  if(!training) {
    data <- obj$data_observed
  }

  if(!is.null(fold)) {
    data <- data %>% filter(cv_fold != fold)
  }

  data <- data[layer$filter(data), ]

  f <- as.formula(formula)
  label <- as.character(terms(f)[[2]])

  if(!is.null(layer$transformation)) {
    data[,label] <- layer$transformation$transform(data[,label])
  }

  data_recipe <- recipe(f, data=data)

  if(layer$method_options$step_log)
    data_recipe <- data_recipe %>% step_log(as.name(label))
  if(layer$method_options$step_normalize)
    data_recipe <- data_recipe %>% step_normalize(all_numeric(), -all_outcomes())

  data_recipe <- data_recipe %>% prep()

  data_baked.glm <- bake(data_recipe, new_data = data)

  if(ncol(data_baked.glm) == 1)
    data_baked.glm <- data_baked.glm %>% mutate(intercept = 1)

  model.glm <- glm(f,data = data_baked.glm, family = layer$method_options$family_for_glm)
  layer$model.glm <- model.glm

  data_recipe <- data_recipe %>% step_dummy(all_nominal(), one_hot = FALSE) %>% prep()

  data_baked <- bake(data_recipe, new_data = data)
  layer$data_recipe <- data_recipe

  if(ncol(data_baked) == 1)
    data_baked <- data_baked %>% mutate(intercept = 1)

  x <- select(data_baked,-as.name(label)) %>% as.matrix()
  y <- data_baked %>% pull(as.name(label))

  layer$x <- x
  layer$y <- y

  inputs <- layer_input(shape = c(length(model.glm$coefficients)-1), name = 'input_layer')

  # GLM Neural network

  GLMNetwork <- inputs %>%
    layer_dense(units=1, activation='linear', name='output_layer_GLM', trainable=FALSE,
                weights=list(array(model.glm$coefficients[2:length(model.glm$coefficients)], dim=c(length(model.glm$coefficients)-1,1)),
                             array(model.glm$coefficients[1],dim=c(1))))

  # Neural network

  NNetwork <- keras_model_sequential(name = 'NNetwork')

  if(layer$method_options$batch_normalization)
    NNetwork <- inputs %>% layer_batch_normalization()

  if(!is.null(layer$method_options$hidden)) {

    n <- length(layer$method_options$hidden)

    for(i in seq(from = 1, to=n)) {
      if(i==1 & !layer$method_options$batch_normalization) {
        NNetwork <- inputs %>%
          layer_dense(units = layer$method_options$hidden[i],
                      name = ifelse(i==n,'last_hidden_layer',paste0('hidden_layer_',i))) %>%
          layer_activation(activation = layer$method_options$activation.hidden[i])
        if(!is.null(layer$method_options$dropout.hidden))
          NNetwork <- NNetwork %>% layer_dropout(rate = layer$method_options$dropout.hidden[i])
      }
      else {
        NNetwork <- NNetwork %>% layer_dense(units = layer$method_options$hidden[i],
                                         name = ifelse(i==n,'last_hidden_layer',paste0('hidden_layer_',i))) %>%
          layer_activation(activation = layer$method_options$activation.hidden[i])
        if(!is.null(layer$method_options$dropout.hidden))
          NNetwork <- NNetwork %>% layer_dropout(rate = layer$method_options$dropout.hidden[i])
      }
    }

  }

  if(!layer$method_options$batch_normalization & is.null(layer$method_options$hidden))
    NNetwork <- inputs %>% layer_dense(units = 1, activation = layer$method_options$activation.output,
                                     use_bias = layer$method_options$use_bias,
                                     name = 'output_layer')
  else
    NNetwork <- NNetwork %>% layer_dense(units = 1, activation = layer$method_options$activation.output,
                                     use_bias = layer$method_options$use_bias,
                                     name = 'output_layer')

  # CANN

  CANNoutput <- list(GLMNetwork, NNetwork) %>% layer_add() %>%
    layer_dense(units = 1, activation = layer$method_options$activation.output.cann, trainable = !layer$method_options$fixed.cann,
                weights = switch(layer$method_options$fixed.cann + 1,NULL,list(array(c(1), dim=c(1,1)),
                               array(0, dim=c(1)))),
                name = 'output_layer_CANN')

  CANN <- keras_model(inputs = inputs, outputs = c(CANNoutput), name = 'CANN')

  print(summary(CANN))

  CANN %>% compile(
    loss = layer$method_options$loss,
    optimizer = layer$method_options$optimizer,
    metrics = layer$method_options$metrics
  )

  earlystopping <- callback_early_stopping(
    monitor = layer$method_options$monitor,
    patience = layer$method_options$patience)

  layer$history <- CANN %>%
    keras::fit(x, y, epochs = layer$method_options$epochs,
               batch_size = layer$method_options$batch_size,
               validation_split = layer$method_options$validation_split,
               callbacks = list(earlystopping))

  if(!layer$method_options$bias_regularization) {
    layer$fit <- CANN
  }
  else {
    # We keep track of the neural network (biased) model
    layer$fit.biased <- CANN
    # Source: Ferrario, Andrea and Noll, Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April 23, 2020), p.52
    glm.formula <- function(nb) {
      string <- "yy ~ X1 "
      if(nb>1) {for (i in 2:nb) {string <- paste(string,"+ X",i, sep="")}}
      string
    }

    zz        <- keras_model(inputs = CANN$input, outputs=get_layer(CANN,'last_hidden_layer')$output)
    layer$zz  <- zz

    Zlearn    <- data.frame(zz %>% predict(x))
    names(Zlearn) <- paste0('X', 1:ncol(Zlearn))
    # We keep track of the pre-processed data for analysis purposes
    layer$Zlearn <- Zlearn

    Zlearn$yy <- y
    if(layer$method_options$distribution == 'gamma')
      fam <- Gamma(link=log) # default link=inverse but we use exponential as activation function
    else if(layer$method_options$distribution == 'bernoulli')
      fam <- binomial() # default link = logit <-> activation function = sigmoid
    else if(layer$method_options$distribution == 'poisson')
      fam <- poisson() # default link = log <-> activation function = exponential
    else if(layer$method_options$distribution == 'gaussian')
      fam <- gaussian() # default link = identity <-> activation function = identity
    else
      stop('Bias regularization is not supported for this distribution.')

    glm1 <- glm(as.formula(glm.formula(ncol(Zlearn)-1)), data=Zlearn, family=fam)
    layer$fit <- glm1
  }

  if(layer$method_options$bias_regularization)
    pred <- layer$fit$fitted.values
  else
    pred <- layer$fit %>% predict(x)

  if(layer$method_options$distribution == 'gaussian') {
    layer$sigma <- sd(pred - y)
  }

  if(layer$method_options$distribution == 'gamma') {
    shape <- hirem_gamma_shape(observed = y, fitted = pred)
    layer$shape <- shape$shape
    layer$shape.se <- shape$se
  }

  return(layer)
}

#' @export
fit.layer_aml_h2o <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat("Fitting layer_aml_h2o ...\n")
  layer$formula <- formula

  data <- obj$data_training
  if(!training) {
    data <- obj$data_observed
  }

  if(!is.null(fold)) {
    data <- data %>% filter(cv_fold != fold)
  }

  data <- data[layer$filter(data), ]

  f <- as.formula(formula)
  label <- as.character(terms(f)[[2]])

  if(!is.null(layer$transformation)) {
    data[,label] <- layer$transformation$transform(data[,label])
  }

  h2o.init()
  h2o.no_progress()

  data.h2o <- as.h2o(data)
  layer$fit <- h2o.automl(x = attr(terms(f),"term.labels"),
                          y = label,
                          training_frame = data.h2o,
                          max_models = layer$method_options$max_models)

  lb <- layer$fit@leaderboard
  print(lb, n = nrow(lb))

  if(layer$method_options$distribution == 'gaussian') {
    layer$sigma <- sd(h2o.predict(layer$fit, data.h2o) - data[,label])
  }

  if(layer$method_options$distribution == 'gamma') {
    shape <- hirem_gamma_shape(data[,label], h2o.predict(layer$fit, data.h2o))
    layer$shape <- shape$shape
    layer$shape.se <- shape$se
  }


  return(layer)
}

#' Fitting layers in a hierarchical reserving model
#'
#' fit one or multiple layers of the hierarchical reserving model
#'
#' @param obj The hierarchical reserving model
#' @param training TRUE: Fit the layers on the training data set \cr
#'                 FALSE: Fit the layers on the observed data set
#' @param fold Index of the fold on which the model should be estimated \cr
#'             When \code{fold == NULL} the layer is estimated on all available records
#' @param weights Optional. Vector of weights per development year since reporting.
#' @param weight.var Optional. The name of the variable representing the development year since reporting in the data set.
#' @param balance.var Optional. The name of the variable representing the development year since reporting in the data set, in case you want to perform a balance correction step.
#' @param ... Add for each layer an argument with the same name as the layer and as value a formula describing the regression model
#'
#' @importFrom dplyr filter %>%
#' @import gbm
#'
#' @export
fit.hirem <- function(obj, training = FALSE, fold = NULL, weights = NULL, weight.var = NULL, balance.var = NULL, ...) {
  formulas <- list(...)
  obj$balance.var <- balance.var
  obj$weight.var  <- weight.var
  obj$weights     <- weights

  for(i in 1:length(formulas)) {
    index <- hirem_get_layer_pos(obj, names(formulas)[i])

    obj$layers[[index]] <- fit(obj$layers[[index]], obj, formulas[[i]], training, fold)
  }

  return(obj)
}

