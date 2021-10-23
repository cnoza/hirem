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
    layer$shape.sd <- shape$se
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
    shape <- gamma_fit_shape(as.matrix(data[,label]), predict(layer$fit, ntreelimit = obj$best_ntreelimit, newdata = data.xgb, type = "response"))
    layer$shape <- shape$shape
    layer$shape_sd <- shape$s.e.
  }

  return(layer)
}

#' @export
fit.layer_mlp <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat("Fitting layer_mlp ...\n")
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
    shape <- gamma_fit_shape(data[,label], h2o.predict(layer$fit, data.h2o))
    layer$shape <- shape$shape
    layer$shape_sd <- shape$s.e.
  }


  return(layer)
}

#' @export
fit.layer_aml <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat("Fitting layer_aml ...\n")
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
    shape <- gamma_fit_shape(data[,label], h2o.predict(layer$fit, data.h2o))
    layer$shape <- shape$shape
    layer$shape_sd <- shape$s.e.
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

