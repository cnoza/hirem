#' @export
fit <- function(object, ...) {
  UseMethod("fit")
}

fit.layer_glm <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat(sprintf("Fitting layer_glm for %s...\n", layer$name))

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
  cat(sprintf("Fitting layer_gbm for %s...\n", layer$name))

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

#' @importFrom xgboost xgb.DMatrix xgb.cv xgb.train
#' @import Matrix
#' @import ParBayesianOptimization
#' @export
fit.layer_xgb <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat(sprintf("Fitting layer_xgb for %s...\n", layer$name))

  layer$formula <- formula

  data <- obj$data_training
  if(!training) {
    data <- obj$data_observed
  }

  if(!is.null(fold)) {
    data <- data %>% filter(cv_fold != fold)
  }

  data <- data[layer$filter(data), ]
  f <- as.formula(paste0(formula,'-1'))
  label <- as.character(terms(f)[[2]])

  if(!is.null(layer$transformation)) {
    data[,label] <- layer$transformation$transform(data[,label])
  }

  contrasts.arg <- lapply(data.frame(data[, sapply(data, is.factor)]),contrasts,contrasts = FALSE)
  names(contrasts.arg) <- colnames(data %>% select_if(is.factor))
  layer$data.model.matrix <- sparse.model.matrix(f,data=data,contrasts.arg = contrasts.arg)

  if(!is.null(obj$weights)) {
    weights.vec <- obj$weights[data[[obj$weight.var]]]
    data.xgb <- xgb.DMatrix(data = as.matrix(layer$data.model.matrix),
                            info = list('label' = as.matrix(data[,label]),
                                        'weight' = as.matrix(weights.vec)))
  }
  else {
    data.xgb <- xgb.DMatrix(data = as.matrix(layer$data.model.matrix),
                            info = list('label' = as.matrix(data[,label])))
  }

  if(layer$method_options$gridsearch_cv) {
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
        stratified = layer$method_options$stratified,
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
  else if(layer$method_options$bayesOpt) {

    Folds <- list()
    names <- c()
    for(i in 1:layer$method_options$nfolds) {
      Folds[[i]] <- as.integer(seq(i,nrow(data.xgb),by = layer$method_options$nfolds))
      names[i] <- paste0('Fold',i)
    }
    names(Folds) <- names

    scoringFunction <- function(max_depth = layer$method_options$max_depth,
                                min_child_weight = layer$method_options$min_child_weight,
                                subsample = layer$method_options$subsample,
                                lambda = layer$method_options$lambda,
                                alpha = layer$method_options$alpha,
                                gamma = layer$method_options$gamma,
                                eta = layer$method_options$eta,
                                colsample_bynode = layer$method_options$colsample_bynode,
                                max_delta_step = layer$method_options$max_delta_step,
                                nthread = layer$method_options$nthread,
                                scale_pos_weight = layer$method_options$scale_pos_weight) {

      Pars <- list(
        booster = layer$method_options$booster
        , tree_method = layer$method_options$tree_method
        , grow_policy = layer$method_options$grow_policy
        , max_depth = max_depth
        , min_child_weight = min_child_weight
        , subsample = subsample
        , lambda = lambda
        , alpha = alpha
        , colsample_bynode = colsample_bynode
        , eta = eta
        , gamma = gamma
        , max_delta_step = max_delta_step
        , objective = layer$method_options$objective
        , eval_metric = layer$method_options$eval_metric
        , nthread = layer$method_options$nthread
        , scale_pos_weight = scale_pos_weight
      )

      xgbcv <- xgb.cv(
        params = Pars
        , data = data.xgb
        , nround = layer$method_options$nrounds
        , folds = Folds
        , early_stopping_rounds = layer$method_options$early_stopping_rounds
        , stratified = layer$method_options$stratified
        , verbose = 0
        , maximize = !layer$method_options$bayesOpt.min
      )

      return(list(Score = ifelse(layer$method_options$bayesOpt.min,
                                 -min(xgbcv$evaluation_log[,4]),
                                 max(xgbcv$evaluation_log[,4])),
                  nrounds = xgbcv$best_iteration))

    }

    if(is.null(layer$method_options$bayesOpt_bounds)) {
      bounds <- list(
        min_child_weight = c(0L,1000L)
        , subsample = c(0.75,1)
        , colsample_bynode = c(0.5, 1)
      )
    }
    else
      bounds <- layer$method_options$bayesOpt_bounds

    bounds_names <- as.vector(names(bounds))

    tNoPar <- system.time(
      optObj <- bayesOpt(
        FUN = scoringFunction
        , bounds = bounds
        , initPoints = max(length(bounds)+1,3)
        , iters.n = layer$method_options$bayesOpt_iters_n
        , iters.k = 1
      )
    )

    # cl <- makeCluster(2)
    # registerDoParallel(cl)
    # clusterExport(cl=cl,varlist=c('Folds','data.xgb'),envir=environment())
    # clusterEvalQ(cl,expr= {
    #   library(xgboost)
    # })
    #
    # tWithPar <- system.time(
    #   optObj <- bayesOpt(
    #     FUN = scoringFunction
    #     , bounds = bounds
    #     , initPoints = 10
    #     , iters.n = layer$method_options$bayesOpt_iters_n
    #     , iters.k = 2
    #     , parallel = TRUE
    #   )
    # )
    # stopCluster(cl)
    # registerDoSEQ()

    layer$optObj <- optObj
    best_index <- as.numeric(which.max(optObj$scoreSummary$Score))
    nrounds <- optObj$scoreSummary$nrounds[best_index]

    params = list(
      booster = layer$method_options$booster,
      objective = layer$method_options$objective,
      eval_metric = layer$method_options$eval_metric,
      tree_method = layer$method_options$tree_method,
      grow_policy = layer$method_options$grow_policy,
      eta = ifelse('eta' %in% bounds_names,getBestPars(optObj)$eta,layer$method_options$eta),
      subsample = ifelse('subsample' %in% bounds_names,getBestPars(optObj)$subsample,layer$method_options$subsample),
      colsample_bynode = ifelse('colsample_bynode' %in% bounds_names,getBestPars(optObj)$colsample_bynode,layer$method_options$colsample_bynode),
      max_depth = ifelse('max_depth' %in% bounds_names,getBestPars(optObj)$max_depth,layer$method_options$max_depth),
      max_delta_step = ifelse('max_delta_step' %in% bounds_names,getBestPars(optObj)$max_delta_step,layer$method_options$max_delta_step),
      min_child_weight = ifelse('min_child_weight' %in% bounds_names,getBestPars(optObj)$min_child_weight,layer$method_options$min_child_weight),
      gamma = ifelse('gamma' %in% bounds_names,getBestPars(optObj)$gamma,layer$method_options$gamma),
      lambda = ifelse('lambda' %in% bounds_names,getBestPars(optObj)$lambda,layer$method_options$lambda),
      alpha = ifelse('alpha' %in% bounds_names,getBestPars(optObj)$alpha,layer$method_options$alpha),
      nthread = ifelse('nthread' %in% bounds_names,getBestPars(optObj)$nthread,layer$method_options$nthread),
      scale_pos_weight = ifelse('scale_pos_weight' %in% bounds_names,getBestPars(optObj)$scale_pos_weight,layer$method_options$scale_pos_weight)
    )

  }
  else {

    nrounds = layer$method_options$nrounds
    params = list(
      booster = layer$method_options$booster,
      objective = layer$method_options$objective,
      eval_metric = layer$method_options$eval_metric,
      tree_method = layer$method_options$tree_method,
      grow_policy = layer$method_options$grow_policy,
      eta = layer$method_options$eta,
      nthread = layer$method_options$nthread,
      subsample = layer$method_options$subsample,
      colsample_bynode = layer$method_options$colsample_bynode,
      max_depth = layer$method_options$max_depth,
      max_delta_step = layer$method_options$max_delta_step,
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

  if(!is.null(obj$balance.var)){
    layer$balance.correction <- sapply(data %>% split(data[[obj$balance.var]]),
                                       function(x) {
                                         contrasts.arg <- lapply(data.frame(x[, sapply(x, is.factor)]),contrasts,contrasts = FALSE)
                                         names(contrasts.arg) <- colnames(x %>% select_if(is.factor))
                                         if(!is.null(obj$weights)) {
                                           weights.vec <- obj$weights[x[[obj$weight.var]]]
                                           newdata <- xgb.DMatrix(data = as.matrix(sparse.model.matrix(f,data=x,contrasts.arg = contrasts.arg)),
                                                                  info = list('label' = as.matrix(x[,label]),'weight' = as.matrix(weights.vec)))
                                         }
                                         else {
                                           newdata <- xgb.DMatrix(data = as.matrix(sparse.model.matrix(f,data=x,contrasts.arg = contrasts.arg)),
                                                                  info = list('label' = as.matrix(x[,label])))
                                         }
                                         sum(x[[layer$name]])/sum(predict(layer$fit,newdata = newdata,ntreelimit = layer$best_ntreelimit,type = 'response'))
                                         }
                                       )
  }

  if(layer$method_options$objective == 'reg:squarederror') {
    layer$sigma <- sd(predict(layer$fit, ntreelimit = layer$best_ntreelimit, newdata = data.xgb, type = "response") - as.matrix(data[,label]))
  }

  if(layer$method_options$objective == 'reg:gamma') {
    if(is.null(obj$weights)) weights.vec <- NULL
    shape <- hirem_gamma_shape(observed = as.matrix(data[,label]),
                               fitted = predict(layer$fit, ntreelimit = layer$best_ntreelimit, newdata = data.xgb, type = "response"),
                               weight = weights.vec)
    layer$shape <- shape$shape
    layer$shape.se <- shape$se
  }

  return(layer)
}

#' @importFrom h2o h2o.init h2o.no_progress as.h2o h2o.deeplearning h2o.predict
#' @export
fit.layer_dnn_h2o <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat(sprintf("Fitting layer_dnn for %s...\n", layer$name))

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

#' @import tensorflow
#' @import keras
#' @importFrom tidyselect all_of
#' @importFrom data.table fsetdiff as.data.table
#' @importFrom recipes recipe step_log step_normalize step_dummy bake prep all_nominal all_numeric all_outcomes
#' @export
fit.layer_dnn <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat(sprintf("Fitting layer_dnn for %s...\n", layer$name))

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

  weights.vec <- if(is.null(obj$weights) | !is.null(layer$method_options$ae.hidden)) NULL else obj$weights[data[[obj$weight.var]]]
  layer$weights.vec <- weights.vec

  data_recipe <- recipe(f, data=data)

  if(layer$method_options$step_log)
    data_recipe <- data_recipe %>% step_log(as.name(label))
  if(layer$method_options$step_normalize)
    data_recipe <- data_recipe %>% step_normalize(all_numeric(), -all_outcomes())
  if(!layer$method_options$use_embedding)
    data_recipe <- data_recipe %>% step_dummy(all_nominal(), one_hot = TRUE)

  data_recipe <- data_recipe %>% prep()
  layer$data_recipe <- data_recipe

  data_baked <- bake(data_recipe, new_data = data)
  if(ncol(data_baked) == 1)
    data_baked <- data_baked %>% mutate(intercept = 1)

  def_x <- def_x_mlp(layer$method_options$use_embedding,
                     layer$method_options$embedding_var,
                     f,
                     data,
                     data_baked,
                     data_recipe,
                     label)

  x       <- def_x$x
  layer$x <- x

  if(layer$method_options$use_embedding) {
    layer$x_fact         <- def_x$x_fact
    layer$x_no_fact      <- def_x$x_no_fact
    layer$fact_var       <- def_x$fact_var
    layer$no_fact_var    <- def_x$no_fact_var
  }

  y <- data_baked %>% pull(as.name(label))

  layer$y <- y

  if(layer$method_options$bayesOpt) {

    Folds <- list()
    names <- c()
    for(i in 1:layer$method_options$nfolds) {
      Folds[[i]] <- as.integer(seq(i,nrow(x),by = layer$method_options$nfolds))
      names[i] <- paste0('Fold',i)
    }
    names(Folds) <- names

    scoringFunction <- function(ae_hidden_1, ae_hidden_2, ae_hidden_3,
                                mlp_hidden_1, mlp_hidden_2, mlp_hidden_3,
                                mlp_dropout.hidden_1, mlp_dropout.hidden_2, mlp_dropout.hidden_3) {

      if(!layer$method_options$use_embedding) {

        ae.hidden <- c()

        if(!is.null(layer$method_options$ae.hidden))
          ae.hidden <- layer$method_options$ae.hidden

        if(!missing(ae_hidden_1)) ae.hidden[1] <- ae_hidden_1
        if(!missing(ae_hidden_2)) ae.hidden[2] <- ae_hidden_2
        if(!missing(ae_hidden_3)) ae.hidden[3] <- ae_hidden_3

        if(length(ae.hidden)==0) ae.hidden <- NULL

        inputs <- layer_input(shape = c(ncol(x)), name='input_layer')

        if(!is.null(ae.hidden)) {

          ae_arch <- def_ae_arch(inputs=inputs,
                                 x=x,
                                 ae.hidden=ae.hidden,
                                 ae.activation.hidden=layer$method_options$ae.activation.hidden)

          autoencoder <- keras_model(inputs, ae_arch$ae_output_l)

          model_en <- keras_model(inputs, ae_arch$ae_hidden_l[[length(ae.hidden)]])

          autoencoder %>% compile(loss = 'mae', optimizer='adam')

          autoencoder %>% keras::fit(
            x = x,
            y = x,
            epochs = layer$method_options$epochs,
            batch_size = layer$method_options$batch_size,
            verbose = layer$method_options$verbose
          )

          x <- model_en %>% predict(x)

          inputs <- layer_input(shape = c(ncol(x)), name='ae_input_layer')

          x.sav <- x

        }

      }

      score <- c()

      for(k in 1:layer$method_options$nfolds) {

        if(!layer$method_options$use_embedding) {
          if(!is.null(ae.hidden))
            x.val    <- x.sav[Folds[[k]],]
          else
            x.val    <- layer$x[Folds[[k]],]
        }
        else {
          x_fact.val <- NULL
          if(length(layer$fact_var)>0) {
            x_fact.val <- list()
            for(i in 1:length(layer$fact_var)) {
              x_fact.val[[i]] <- layer$x_fact[[i]][Folds[[k]]] %>% as.integer()
            }
          }

          x_no_fact.val <- NULL
          if(length(layer$no_fact_var)>0)
            x_no_fact.val  <- layer$x_no_fact[Folds[[k]],] %>% as.matrix()
        }

        y.val    <- layer$y[Folds[[k]]]
        sample.w.val <- weights.vec[Folds[[k]]]

        if(layer$method_options$nfolds==1) {
          if(!layer$method_options$use_embedding) {
            x    <- x.val
          }
          else {
            x_fact        <- x_fact.val
            x_no_fact     <- x_no_fact.val
          }
          y    <- y.val
          sample.w <- sample.w.val
        }
        else {

          if(!layer$method_options$use_embedding) {
            if(!is.null(ae.hidden))
              x <- x.sav[-Folds[[k]],]
            else
              x <- layer$x[-Folds[[k]],]
          }
          else {
            x_fact <- NULL
            if(length(layer$fact_var)>0) {
              x_fact <- list()
              for(i in 1:length(layer$fact_var)) {
                x_fact[[i]] <- layer$x_fact[[i]][-Folds[[k]]] %>% as.integer()
              }
            }
            x_no_fact <- NULL
            if(length(layer$no_fact_var)>0)
              x_no_fact <- layer$x_no_fact[-Folds[[k]],] %>% as.matrix()
          }

          y <- layer$y[-Folds[[k]]]
          sample.w <- weights.vec[-Folds[[k]]]

        }

        mlp_hidden <- c()

        if(!is.null(layer$method_options$hidden))
          mlp_hidden <- layer$method_options$hidden

        if(!missing(mlp_hidden_1)) mlp_hidden[1] <- mlp_hidden_1
        if(!missing(mlp_hidden_2)) mlp_hidden[2] <- mlp_hidden_2
        if(!missing(mlp_hidden_3)) mlp_hidden[3] <- mlp_hidden_3

        if(length(mlp_hidden)==0) mlp_hidden <- NULL

        mlp_dropout.hidden <- c()

        if(!is.null(layer$method_options$dropout.hidden))
          mlp_dropout.hidden <- layer$method_options$dropout.hidden
        else {
          if(!is.null(mlp_hidden))
            mlp_dropout.hidden <- rep(0,length(mlp_hidden))
        }

        if(!missing(mlp_dropout.hidden_1)) mlp_dropout.hidden[1] <- mlp_dropout.hidden_1
        if(!missing(mlp_dropout.hidden_2)) mlp_dropout.hidden[2] <- mlp_dropout.hidden_2
        if(!missing(mlp_dropout.hidden_3)) mlp_dropout.hidden[3] <- mlp_dropout.hidden_3

        if(length(mlp_dropout.hidden)==0) mlp_dropout.hidden <- NULL

        def_inputs <- def_inputs_mlp(use_embedding=layer$method_options$use_embedding,
                                     x=x,
                                     no_fact_var=layer$no_fact_var,
                                     fact_var=layer$fact_var,
                                     x_fact=x_fact,
                                     output_dim = layer$method_options$output_dim)

        mlp_arch <- def_mlp_arch(inputs=def_inputs$inputs,
                                 batch_normalization=layer$method_options$batch_normalization,
                                 hidden=mlp_hidden,
                                 activation.hidden=layer$method_options$activation.hidden,
                                 dropout.hidden=mlp_dropout.hidden,
                                 family_for_init=layer$method_options$family_for_init,
                                 label=label,
                                 data=data,
                                 activation.output=layer$method_options$activation.output,
                                 x=x,
                                 use_bias=layer$method_options$use_bias,
                                 weights.vec=weights.vec)

        if(!layer$method_options$use_embedding) {
          model <- keras_model(inputs = def_inputs$inputs, outputs = c(mlp_arch$output))
        }
        else {
          model <- keras_model(inputs = c(def_inputs$inputs_no_fact,
                                          def_inputs$input_layer_emb),
                               outputs = c(mlp_arch$output))
        }

        model %>% compile(
          loss = layer$method_options$loss,
          optimizer = layer$method_options$optimizer,
          metrics = layer$method_options$metrics
        )

        earlystopping <- callback_early_stopping(
          monitor = layer$method_options$monitor,
          patience = layer$method_options$patience)

        if(!layer$method_options$use_embedding) {
          x.inputs <- list(x)
          x.inputs.val <- list(x.val)
        }
        else {
          x.inputs <- list(x_no_fact,x_fact)
          x.inputs[sapply(x.inputs, is.null)] <- NULL
          x.inputs.val <- list(x_no_fact.val,x_fact.val)
          x.inputs.val[sapply(x.inputs.val, is.null)] <- NULL
        }

        history <- model %>%
          keras::fit(x=x.inputs, y=y, sample_weight = sample.w, epochs = layer$method_options$epochs,
                     batch_size = layer$method_options$batch_size,
                     #validation_split = layer$method_options$validation_split,
                     validation_data = list(x.inputs.val,y.val,sample.w.val),
                     callbacks = list(earlystopping),
                     verbose = layer$method_options$verbose)

        score[k] <- ifelse(layer$method_options$bayesOpt.min, -min(history$metrics[[2]]), max(history$metrics[[2]]))

      }

      return(list(Score = mean(score),
                  Pred = 0))

    }


    bounds <- layer$method_options$bayesOpt_bounds

    bounds_names <- as.vector(names(bounds))

    tNoPar <- system.time(
      optObj <- bayesOpt(
        FUN = scoringFunction
        , bounds = bounds
        , initPoints = max(length(bounds)+1,3)
        , iters.n = layer$method_options$bayesOpt_iters_n
        , iters.k = 1
      )
    )

    # cl <- makeCluster(2)
    # registerDoParallel(cl)
    # clusterExport(cl=cl,varlist=c('Folds','data.xgb'),envir=environment())
    # clusterEvalQ(cl,expr= {
    #   library(xgboost)
    # })
    #
    # tWithPar <- system.time(
    #   optObj <- bayesOpt(
    #     FUN = scoringFunction
    #     , bounds = bounds
    #     , initPoints = 10
    #     , iters.n = layer$method_options$bayesOpt_iters_n
    #     , iters.k = 2
    #     , parallel = TRUE
    #   )
    # )
    # stopCluster(cl)
    # registerDoSEQ()

    layer$optObj <- optObj

    if(is.null(layer$method_options$ae.hidden) &
       (('ae_hidden_1' %in% bounds_names) | ('ae_hidden_2' %in% bounds_names) | ('ae_hidden_3' %in% bounds_names)) ) {
      layer$method_options$ae.hidden <- c()
    }

    if('ae_hidden_1' %in% bounds_names) layer$method_options$ae.hidden[1] <- getBestPars(optObj)$ae_hidden_1
    if('ae_hidden_2' %in% bounds_names) layer$method_options$ae.hidden[2] <- getBestPars(optObj)$ae_hidden_2
    if('ae_hidden_3' %in% bounds_names) layer$method_options$ae.hidden[3] <- getBestPars(optObj)$ae_hidden_3

    if(is.null(layer$method_options$hidden) &
       (('mlp_hidden_1' %in% bounds_names) | ('mlp_hidden_2' %in% bounds_names) | ('mlp_hidden_3' %in% bounds_names)) ) {
      layer$method_options$hidden <- c()
      layer$method_options$bias_regularization <- TRUE
    }

    if('mlp_hidden_1' %in% bounds_names) layer$method_options$hidden[1] <- getBestPars(optObj)$mlp_hidden_1
    if('mlp_hidden_2' %in% bounds_names) layer$method_options$hidden[2] <- getBestPars(optObj)$mlp_hidden_2
    if('mlp_hidden_3' %in% bounds_names) layer$method_options$hidden[3] <- getBestPars(optObj)$mlp_hidden_3

    if(is.null(layer$method_options$dropout.hidden) &
       (('mlp_dropout.hidden_1' %in% bounds_names) | ('mlp_dropout.hidden_2' %in% bounds_names) | ('mlp_dropout.hidden_3' %in% bounds_names)) )
      layer$method_options$dropout.hidden <- c()

    if('mlp_dropout.hidden_1' %in% bounds_names) layer$method_options$dropout.hidden[1] <- getBestPars(optObj)$mlp_dropout.hidden_1
    if('mlp_dropout.hidden_2' %in% bounds_names) layer$method_options$dropout.hidden[2] <- getBestPars(optObj)$mlp_dropout.hidden_2
    if('mlp_dropout.hidden_3' %in% bounds_names) layer$method_options$dropout.hidden[3] <- getBestPars(optObj)$mlp_dropout.hidden_3

  }

  x <- layer$x

  if(layer$method_options$use_embedding) {
    x_fact        <-layer$x_fact
    x_no_fact     <-layer$x_no_fact
    no_fact_var   <-layer$no_fact_var
    fact_var      <- layer$fact_var
  }

  y <- layer$y
  sample.w <- weights.vec

  if(!is.null(layer$method_options$ae.hidden)) {

    ae_arch <- def_ae_arch(inputs=inputs,
                           x=x,
                           ae.hidden=layer$method_options$ae.hidden,
                           ae.activation.hidden=layer$method_options$ae.activation.hidden)

    autoencoder <- keras_model(inputs, ae_arch$ae_output_l)

    model_en <- keras_model(inputs, ae_arch$ae_hidden_l[[length(layer$method_options$ae.hidden)]])

    # Autoencoder model
    summary(autoencoder)
    # Encoder model
    summary(model_en)

    autoencoder %>% compile(loss = 'mae', optimizer='adam')

    autoencoder %>% keras::fit(
      x = x,
      y = x,
      epochs = layer$method_options$epochs,
      batch_size = layer$method_options$batch_size,
      verbose = layer$method_options$verbose
    )

    x <- model_en %>% predict(x)
    inputs <- layer_input(shape = c(ncol(x)), name='ae_input_layer')
    layer$x.encoded <- x
    layer$model_en <- model_en

  }

  def_inputs <- def_inputs_mlp(use_embedding=layer$method_options$use_embedding,
                               x=x,
                               no_fact_var=no_fact_var,
                               fact_var=fact_var,
                               x_fact=x_fact,
                               output_dim = layer$method_options$output_dim)

  mlp_arch <- def_mlp_arch(inputs=def_inputs$inputs,
                           batch_normalization=layer$method_options$batch_normalization,
                           hidden=layer$method_options$hidden,
                           activation.hidden=layer$method_options$activation.hidden,
                           dropout.hidden=layer$method_options$dropout.hidden,
                           family_for_init=layer$method_options$family_for_init,
                           label=label,
                           data=data,
                           activation.output=layer$method_options$activation.output,
                           x=x,
                           use_bias=layer$method_options$use_bias,
                           weights.vec=weights.vec)

  if(!is.null(layer$method_options$family_for_init)) layer$glm.hom <- mlp_arch$glm.hom

  if(!layer$method_options$use_embedding) {
    model <- keras_model(inputs = def_inputs$inputs, outputs = c(mlp_arch$output))
  }
  else {
    model <- keras_model(inputs = c(def_inputs$inputs_no_fact,
                                    def_inputs$input_layer_emb),
                         outputs = c(mlp_arch$output))
  }

  print(summary(model))

  model %>% compile(
    loss = layer$method_options$loss,
    optimizer = layer$method_options$optimizer,
    metrics = layer$method_options$metrics
  )

  earlystopping <- callback_early_stopping(
    monitor = layer$method_options$monitor,
    patience = layer$method_options$patience)

  if(!layer$method_options$use_embedding) {
    x.inputs <- list(x)
  }
  else {
    x.inputs <- list(x_no_fact,x_fact)
    x.inputs[sapply(x.inputs, is.null)] <- NULL
  }

  now <- Sys.time()
  fn <- paste0("dnn_best_weights_",format(now, "%Y%m%d_%H%M%S.hdf5"))

  CBs <- callback_model_checkpoint(fn, monitor=layer$method_options$monitor, save_best_only = TRUE, save_weights_only = TRUE)

  layer$history <- model %>%
    keras::fit(x=x.inputs, y=y, sample_weight=sample.w, epochs = layer$method_options$epochs,
               batch_size = layer$method_options$batch_size,
               validation_split = layer$method_options$validation_split,
               callbacks = list(earlystopping, CBs),
               verbose = layer$method_options$verbose)

  load_model_weights_hdf5(model, fn)

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

    glm.formula.2 <- function(cov) {
      string <- "yy ~ "
      for (i in 2:length(cov)) {
        if(i==2) string <- paste0(string,' ',cov[i])
        else string <- paste0(string,' + ',cov[i])
      }
      string
    }

    zz        <- keras_model(inputs = model$input, outputs=get_layer(model,'last_hidden_layer_activation')$output)
    layer$zz  <- zz

    if(!layer$method_options$use_embedding) {
      x <- def_x$x
      x.inputs <- list(x)
    }
    else {
      x_fact        <- def_x$x_fact
      x_no_fact     <- def_x$x_no_fact
      x.inputs      <- list(x_no_fact,x_fact)
    }

    Zlearn    <- data.frame(zz %>% predict(x.inputs))
    names(Zlearn) <- paste0('X', 1:ncol(Zlearn))

    Zlearn$yy <- y

    # We keep track of the pre-processed data for analysis purposes
    layer$Zlearn <- Zlearn

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

    glm1 <- glm(as.formula(glm.formula(ncol(Zlearn)-1)), data=Zlearn, family=fam, weights = weights.vec)
    cov <- names(glm1$coefficients[!sapply(glm1$coefficients,is.na)])
    if(length(cov)>0) {
      glm2 <- glm(as.formula(glm.formula.2(cov)), data=Zlearn, family=fam, weights = weights.vec)
      layer$fit <- glm2
    }
    else layer$fit <- glm1
  }

  if(!is.null(obj$balance.var)){
      layer$balance.correction <- sapply(data %>% group_split(data[[obj$balance.var]]),
                                         function(x) {
                                           data_baked_bc <- bake(layer$data_recipe, new_data = x)
                                           if(ncol(data_baked_bc) == 1)
                                             data_baked_bc <- data_baked_bc %>% mutate(intercept = 1)

                                           def_x <- def_x_mlp(layer$method_options$use_embedding,
                                                              layer$method_options$embedding_var,
                                                              f,
                                                              x,
                                                              data_baked_bc,
                                                              layer$data_recipe,
                                                              label)

                                           if(!layer$method_options$use_embedding) {
                                             x.tmp <- def_x$x
                                             x.inputs.tmp <- list(x.tmp)
                                           }
                                           else {
                                             x_fact.tmp         <- def_x$x_fact
                                             x_no_fact.tmp      <- def_x$x_no_fact
                                             x.inputs.tmp <- list(x_no_fact.tmp,x_fact.tmp)
                                             x.inputs.tmp[sapply(x.inputs.tmp, is.null)] <- NULL
                                           }

                                           if(!is.null(layer$method_options$ae.hidden))
                                             x.tmp <- model_en %>% predict(x.tmp)

                                           if(layer$method_options$bias_regularization) {
                                             Zlearn.tmp   <- data.frame(layer$zz %>% predict(x.inputs.tmp))
                                             names(Zlearn.tmp) <- paste0('X', 1:ncol(Zlearn.tmp))
                                             sum(x[[layer$name]])/sum(predict(layer$fit, newdata = Zlearn.tmp, type = 'response'))
                                           }
                                           else {
                                             sum(x[[layer$name]])/sum(layer$fit %>% predict(x.inputs.tmp))
                                           }
                                        })
  }

  if(layer$method_options$bias_regularization)
    pred <- layer$fit$fitted.values
  else
    pred <- layer$fit %>% predict(x.inputs)

  if(layer$method_options$distribution == 'gaussian') {
    layer$sigma <- sd(pred - y)
  }

  if(layer$method_options$distribution == 'gamma') {
    shape <- hirem_gamma_shape(observed = y, fitted = pred, weight = weights.vec)
    layer$shape <- shape$shape
    layer$shape.se <- shape$se
  }

  return(layer)
}

#' @importFrom data.table transpose
#' @export
fit.layer_cann <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat(sprintf("Fitting layer_cann for %s...\n", layer$name))

  layer$formula <- formula

  data <- obj$data_training
  if(!training) {
    data <- obj$data_observed
  }

  if(!is.null(fold)) {
    data <- data %>% filter(cv_fold != fold)
  }

  data <- data[layer$filter(data), ]

  f     <- as.formula(formula)
  label <- as.character(terms(f)[[2]])

  if(!is.null(layer$method_options$formula.glm))
    f.glm <- as.formula(layer$method_options$formula.glm)
  else
    f.glm <- f

  if(!is.null(layer$transformation)) {
    data[,label] <- layer$transformation$transform(data[,label])
  }

  if(!is.null(obj$weights)) weights.vec <- obj$weights[data[[obj$weight.var]]] else weights.vec <- NULL

  data_recipe     <- recipe(f, data=data)
  data_recipe.glm <- recipe(f.glm, data=data)

  if(layer$method_options$step_log) {
    data_recipe     <- data_recipe %>% step_log(as.name(label))
    data_recipe.glm <- data_recipe.glm %>% step_log(as.name(label))
  }
  if(layer$method_options$step_normalize) {
    data_recipe     <- data_recipe %>% step_normalize(all_numeric(), -all_outcomes())
    data_recipe.glm <- data_recipe.glm %>% step_normalize(all_numeric(), -all_outcomes())
  }

  data_recipe     <- data_recipe %>% prep()
  data_recipe.glm <- data_recipe.glm %>% prep()
  layer$data_recipe.glm.no_dummy <- data_recipe.glm

  data_baked.glm <- bake(data_recipe.glm, new_data = data)

  if(ncol(data_baked.glm) == 1)
    data_baked.glm <- data_baked.glm %>% mutate(intercept = 1)

  model.glm       <- glm(f.glm, data = data_baked.glm, family = layer$method_options$family_for_glm, weights = weights.vec)
  layer$model.glm <- model.glm

  if(!layer$method_options$use_embedding) {
    data_recipe.glm <- data_recipe.glm %>% step_dummy(all_nominal(), one_hot = FALSE) %>% prep()
    data_recipe     <- data_recipe %>% step_dummy(all_nominal(), one_hot = TRUE) %>% prep()
  }

  data_baked.glm <- bake(data_recipe.glm, new_data = data)
  data_baked <- bake(data_recipe, new_data = data)
  nrows <- nrow(data_baked)

  if(ncol(data_baked) == 1)
    data_baked <- data_baked %>% mutate(intercept = 1)

  if(ncol(data_baked.glm) == 1)
    data_baked.glm <- data_baked.glm %>% mutate(intercept = 1)

  layer$data_recipe.glm <- data_recipe.glm
  layer$data_recipe     <- data_recipe

  def_x <- def_x(layer$method_options$use_embedding,
                 layer$method_options$embedding_var,
                 layer$method_options$embedding_var.glm,
                 f,
                 f.glm,
                 data,
                 data_baked,
                 data_baked.glm,
                 data_recipe,
                 data_recipe.glm,
                 label)

  layer$x     <- def_x$x
  layer$x.glm <- def_x$x.glm

  if(layer$method_options$use_embedding) {
    layer$x_fact         <- def_x$x_fact
    layer$x_no_fact      <- def_x$x_no_fact
    layer$x_fact.glm     <- def_x$x_fact.glm
    layer$x_no_fact.glm  <- def_x$x_no_fact.glm
    layer$fact_var       <- def_x$fact_var
    layer$no_fact_var    <- def_x$no_fact_var
    layer$fact_var.glm   <- def_x$fact_var.glm
    layer$no_fact_var.glm  <- def_x$no_fact_var.glm
  }

  y       <- data_baked %>% pull(as.name(label))
  layer$y <- y

  if(layer$method_options$bayesOpt) {

    Folds <- list()
    names <- c()
    for(i in 1:layer$method_options$nfolds) {
      Folds[[i]] <- as.integer(seq(i,nrows,by = layer$method_options$nfolds))
      names[i]   <- paste0('Fold',i)
    }
    names(Folds) <- names

    scoringFunction <- function(mlp_hidden_1, mlp_hidden_2, mlp_hidden_3,
                                mlp_dropout.hidden_1, mlp_dropout.hidden_2, mlp_dropout.hidden_3) {

      score <- c()

      for(k in 1:layer$method_options$nfolds) {

        if(!layer$method_options$use_embedding) {
          x.val         <- layer$x[Folds[[k]],] %>% as.matrix()
          x.glm.val     <- layer$x.glm[Folds[[k]],] %>% as.matrix()
        }
        else {

          x_fact.val <- NULL
          if(length(layer$fact_var)>0) {
            x_fact.val <- list()
            for(i in 1:length(layer$fact_var)) {
              x_fact.val[[i]] <- layer$x_fact[[i]][Folds[[k]]] %>% as.integer()
            }
          }

          x_no_fact.val <- NULL
          if(length(layer$no_fact_var)>0)
            x_no_fact.val  <- layer$x_no_fact[Folds[[k]],] %>% as.matrix()

          x_fact.glm.val <- NULL
          if(length(layer$fact_var.glm)>0) {
            x_fact.glm.val <- list()
            for(i in 1:length(layer$fact_var.glm)) {
              x_fact.glm.val[[i]] <- layer$x_fact.glm[[i]][Folds[[k]]] %>% as.integer()
            }
          }

          x_no_fact.glm.val <- NULL
          if(length(layer$no_fact_var.glm)>0)
            x_no_fact.glm.val <- layer$x_no_fact.glm[Folds[[k]],] %>% as.matrix()

        }

        y.val        <- layer$y[Folds[[k]]]
        sample.w.val <- weights.vec[Folds[[k]]]

        if(layer$method_options$nfolds==1) {

          if(!layer$method_options$use_embedding) {
            x     <- x.val
            x.glm <- x.glm.val
          }
          else {
            x_fact        <- x_fact.val
            x_no_fact     <- x_no_fact.val
            x_fact.glm    <- x_fact.glm.val
            x_no_fact.glm <- x_no_fact.glm.val
          }

          y <- y.val
          sample.w <- sample.w.val

        }
        else {

          if(!layer$method_options$use_embedding) {
            x      <- layer$x[-Folds[[k]],]
            x.glm  <- layer$x.glm[-Folds[[k]],]
          }
          else {
            x_fact <- NULL
            if(length(layer$fact_var)>0) {
              x_fact <- list()
              for(i in 1:length(layer$fact_var)) {
                x_fact[[i]] <- layer$x_fact[[i]][-Folds[[k]]] %>% as.integer()
              }
            }
            x_no_fact <- NULL
            if(length(layer$no_fact_var)>0)
              x_no_fact <- layer$x_no_fact[-Folds[[k]],] %>% as.matrix()

            x_fact.glm <- NULL
            if(length(layer$fact_var.glm)>0) {
              x_fact.glm <- list()
              for(i in 1:length(layer$fact_var.glm)) {
                x_fact.glm[[i]] <- layer$x_fact.glm[[i]][-Folds[[k]]] %>% as.integer()
              }
            }
            x_no_fact.glm <- NULL
            if(length(layer$no_fact_var.glm)>0)
              x_no_fact.glm <- layer$x_no_fact.glm[-Folds[[k]],] %>% as.matrix()
          }

          y <- layer$y[-Folds[[k]]]
          sample.w <- weights.vec[-Folds[[k]]]

        }

        def_inputs <- def_inputs(use_embedding=layer$method_options$use_embedding,
                                 x=x,
                                 model.glm=model.glm,
                                 no_fact_var=layer$no_fact_var,
                                 no_fact_var.glm=layer$no_fact_var.glm,
                                 fact_var.glm=layer$fact_var.glm,
                                 fact_var=layer$fact_var,
                                 x_fact=x_fact,
                                 x_fact.glm=x_fact.glm)

        # GLM Neural network

        if(!layer$method_options$use_embedding) {
          GLMNetwork.tmp <- def_inputs$inputs.glm %>%
            layer_dense(units=1, activation='linear', name='output_layer_glm', trainable=FALSE,
                        weights=list(array(model.glm$coefficients[2:length(model.glm$coefficients)], dim=c(length(model.glm$coefficients)-1,1)),
                                     array(model.glm$coefficients[1],dim=c(1))))
        }
        else {

          coef_no_fact.glm <- NULL
          if(length(def_inputs$beta.no_fact_var.glm) > 1)
            coef_no_fact.glm <- def_inputs$beta.no_fact_var.glm[2:length(def_inputs$beta.no_fact_var.glm)]

          coef_fact_var.glm <- NULL
          if(length(layer$fact_var.glm)>0)
            coef_fact_var.glm <- rep(1,length(layer$fact_var.glm))

          GLMNetwork.tmp <- def_inputs$inputs.glm %>%
            layer_dense(units=1, activation='linear', name='output_layer_glm', trainable=FALSE,
                        weights=list(array(c(coef_no_fact.glm,coef_fact_var.glm),
                                           dim=c(length(layer$fact_var.glm)+length(def_inputs$beta.no_fact_var.glm)-1,1)),
                                     array(def_inputs$beta.no_fact_var.glm[1],dim=c(1))))
        }

        # Hyperparameters

        mlp_hidden <- c()

        if(!is.null(layer$method_options$hidden))
          mlp_hidden <- layer$method_options$hidden

        if(!missing(mlp_hidden_1)) mlp_hidden[1] <- mlp_hidden_1
        if(!missing(mlp_hidden_2)) mlp_hidden[2] <- mlp_hidden_2
        if(!missing(mlp_hidden_3)) mlp_hidden[3] <- mlp_hidden_3

        if(length(mlp_hidden)==0) mlp_hidden <- NULL

        mlp_dropout.hidden <- c()

        if(!is.null(layer$method_options$dropout.hidden))
          mlp_dropout.hidden <- layer$method_options$dropout.hidden
        else {
          if(!is.null(mlp_hidden))
            mlp_dropout.hidden <- rep(0,length(mlp_hidden))
        }

        if(!missing(mlp_dropout.hidden_1)) mlp_dropout.hidden[1] <- mlp_dropout.hidden_1
        if(!missing(mlp_dropout.hidden_2)) mlp_dropout.hidden[2] <- mlp_dropout.hidden_2
        if(!missing(mlp_dropout.hidden_3)) mlp_dropout.hidden[3] <- mlp_dropout.hidden_3

        if(length(mlp_dropout.hidden)==0) mlp_dropout.hidden <- NULL

        # Neural network

        NNetwork.tmp <- def_NN_arch(def_inputs$inputs,
                                    layer$method_options$batch_normalization,
                                    mlp_hidden,
                                    layer$method_options$activation.hidden,
                                    mlp_dropout.hidden,
                                    layer$method_options$activation.output,
                                    layer$method_options$use_bias)

        # CANN

        CANNoutput.tmp <- list(GLMNetwork.tmp, NNetwork.tmp) %>% layer_add() %>%
          layer_dense(units = 1, activation = layer$method_options$activation.output.cann, trainable = !layer$method_options$fixed.cann,
                      weights = switch(layer$method_options$fixed.cann + 1,NULL,list(array(c(1), dim=c(1,1)),
                                                                                     array(0, dim=c(1)))),
                      name = 'output_layer_CANN')

        if(!layer$method_options$use_embedding)
          CANN.tmp <- keras_model(inputs = c(def_inputs$inputs,
                                             def_inputs$inputs.glm),
                                  outputs = c(CANNoutput.tmp), name = 'CANN.tmp')
        else
          CANN.tmp <- keras_model(inputs = c(def_inputs$inputs_no_fact.glm,
                                             def_inputs$input_layer_emb.glm,
                                             def_inputs$inputs_no_fact,
                                             def_inputs$input_layer_emb),
                                  outputs = c(CANNoutput.tmp), name = 'CANN.tmp')

        CANN.tmp %>% compile(
          loss = layer$method_options$loss,
          optimizer = layer$method_options$optimizer,
          metrics = layer$method_options$metrics
        )

        earlystopping <- callback_early_stopping(
          monitor = layer$method_options$monitor,
          patience = layer$method_options$patience)

        if(!layer$method_options$use_embedding) {
          x.inputs <- list(x,x.glm)
          x.inputs.val <- list(x.val,x.glm.val)
        }
        else {
          x.inputs <- list(x_no_fact.glm,x_fact.glm,x_no_fact,x_fact)
          x.inputs[sapply(x.inputs, is.null)] <- NULL
          x.inputs.val <- list(x_no_fact.glm.val,x_fact.glm.val,x_no_fact.val,x_fact.val)
          x.inputs.val[sapply(x.inputs.val, is.null)] <- NULL
        }

        history.tmp <- CANN.tmp %>%
          keras::fit(x=x.inputs, y=y, sample_weight = sample.w, epochs = layer$method_options$epochs,
                     batch_size = layer$method_options$batch_size,
                     #validation_split = layer$method_options$validation_split,
                     validation_data = list(x.inputs.val,y.val,sample.w.val),
                     callbacks = list(earlystopping),
                     verbose = layer$method_options$verbose)

        score[k] <- ifelse(layer$method_options$bayesOpt.min, -min(history.tmp$metrics[[2]]), max(history.tmp$metrics[[2]]))

      }

      return(list(Score = mean(score),
                  Pred = 0))

    }


    bounds <- layer$method_options$bayesOpt_bounds

    bounds_names <- as.vector(names(bounds))

    tNoPar <- system.time(
      optObj <- bayesOpt(
        FUN = scoringFunction
        , bounds = bounds
        , initPoints = max(length(bounds)+1,3)
        , iters.n = layer$method_options$bayesOpt_iters_n
        , iters.k = 1
      )
    )

    # cl <- makeCluster(2)
    # registerDoParallel(cl)
    # clusterExport(cl=cl,varlist=c('Folds','data.xgb'),envir=environment())
    # clusterEvalQ(cl,expr= {
    #   library(xgboost)
    # })
    #
    # tWithPar <- system.time(
    #   optObj <- bayesOpt(
    #     FUN = scoringFunction
    #     , bounds = bounds
    #     , initPoints = 10
    #     , iters.n = layer$method_options$bayesOpt_iters_n
    #     , iters.k = 2
    #     , parallel = TRUE
    #   )
    # )
    # stopCluster(cl)
    # registerDoSEQ()

    layer$optObj <- optObj

    if(is.null(layer$method_options$hidden) &
       (('mlp_hidden_1' %in% bounds_names) | ('mlp_hidden_2' %in% bounds_names) | ('mlp_hidden_3' %in% bounds_names)) ) {
      layer$method_options$hidden <- c()
      #layer$method_options$bias_regularization <- TRUE
    }

    if('mlp_hidden_1' %in% bounds_names) layer$method_options$hidden[1] <- getBestPars(optObj)$mlp_hidden_1
    if('mlp_hidden_2' %in% bounds_names) layer$method_options$hidden[2] <- getBestPars(optObj)$mlp_hidden_2
    if('mlp_hidden_3' %in% bounds_names) layer$method_options$hidden[3] <- getBestPars(optObj)$mlp_hidden_3

    if(is.null(layer$method_options$dropout.hidden) &
       (('mlp_dropout.hidden_1' %in% bounds_names) | ('mlp_dropout.hidden_2' %in% bounds_names) | ('mlp_dropout.hidden_3' %in% bounds_names)) )
      layer$method_options$dropout.hidden <- c()

    if('mlp_dropout.hidden_1' %in% bounds_names) layer$method_options$dropout.hidden[1] <- getBestPars(optObj)$mlp_dropout.hidden_1
    if('mlp_dropout.hidden_2' %in% bounds_names) layer$method_options$dropout.hidden[2] <- getBestPars(optObj)$mlp_dropout.hidden_2
    if('mlp_dropout.hidden_3' %in% bounds_names) layer$method_options$dropout.hidden[3] <- getBestPars(optObj)$mlp_dropout.hidden_3

  }

  x        <- layer$x
  x.glm    <- layer$x.glm

  if(layer$method_options$use_embedding) {
    x_fact        <-layer$x_fact
    x_no_fact     <-layer$x_no_fact
    x_fact.glm    <-layer$x_fact.glm
    x_no_fact.glm <-layer$x_no_fact.glm
    no_fact_var   <-layer$no_fact_var
    no_fact_var.glm <- layer$no_fact_var.glm
    fact_var.glm  <- layer$fact_var.glm
    fact_var      <- layer$fact_var
  }

  y        <- layer$y
  sample.w <- weights.vec


  def_inputs <- def_inputs(use_embedding=layer$method_options$use_embedding,
                           x=x,
                           model.glm=model.glm,
                           no_fact_var=no_fact_var,
                           no_fact_var.glm=no_fact_var.glm,
                           fact_var.glm=fact_var.glm,
                           fact_var=fact_var,
                           x_fact=x_fact,
                           x_fact.glm=x_fact.glm)

  # GLM Neural network

  if(!layer$method_options$use_embedding) {
    GLMNetwork <- def_inputs$inputs.glm %>%
      layer_dense(units=1, activation='linear', name='output_layer_glm', trainable=FALSE,
                  weights=list(array(model.glm$coefficients[2:length(model.glm$coefficients)], dim=c(length(model.glm$coefficients)-1,1)),
                               array(model.glm$coefficients[1],dim=c(1))))
  }
  else {
    coef_no_fact.glm <- NULL
    if(length(def_inputs$beta.no_fact_var.glm) > 1)
      coef_no_fact.glm <- def_inputs$beta.no_fact_var.glm[2:length(def_inputs$beta.no_fact_var.glm)]

    coef_fact_var.glm <- NULL
    if(length(fact_var.glm)>0)
      coef_fact_var.glm <- rep(1,length(fact_var.glm))

    GLMNetwork <- def_inputs$inputs.glm %>%
      layer_dense(units=1, activation='linear', name='output_layer_glm', trainable=FALSE,
                  weights=list(array(c(coef_no_fact.glm,coef_fact_var.glm),
                                     dim=c(length(fact_var.glm)+length(def_inputs$beta.no_fact_var.glm)-1,1)),
                               array(def_inputs$beta.no_fact_var.glm[1],dim=c(1))))
  }

  # Neural network

  NNetwork <- def_NN_arch(inputs=def_inputs$inputs,
                          batch_normalization=layer$method_options$batch_normalization,
                          hidden=layer$method_options$hidden,
                          activation.hidden=layer$method_options$activation.hidden,
                          dropout.hidden=layer$method_options$dropout.hidden,
                          activation.output=layer$method_options$activation.output,
                          use_bias=layer$method_options$use_bias)

  # CANN

  CANNoutput <- list(GLMNetwork, NNetwork) %>% layer_add() %>%
    layer_dense(units = 1, activation = layer$method_options$activation.output.cann, trainable = !layer$method_options$fixed.cann,
                weights = switch(layer$method_options$fixed.cann + 1,NULL,list(array(c(1), dim=c(1,1)),
                                                                               array(0, dim=c(1)))),
                name = 'output_layer_CANN')

  if(!layer$method_options$use_embedding)
    CANN <- keras_model(inputs = c(def_inputs$inputs,
                                   def_inputs$inputs.glm),
                        outputs = c(CANNoutput), name = 'CANN')
  else
    CANN <- keras_model(inputs = c(def_inputs$inputs_no_fact.glm,
                                   def_inputs$input_layer_emb.glm,
                                   def_inputs$inputs_no_fact,
                                   def_inputs$input_layer_emb),
                        outputs = c(CANNoutput), name = 'CANN')

  print(summary(CANN))

  CANN %>% compile(
    loss = layer$method_options$loss,
    optimizer = layer$method_options$optimizer,
    metrics = layer$method_options$metrics
  )

  earlystopping <- callback_early_stopping(
    monitor = layer$method_options$monitor,
    patience = layer$method_options$patience)

  if(!layer$method_options$use_embedding) {
    x.inputs <- list(x,x.glm)
  }
  else {
    x.inputs <- list(x_no_fact.glm,x_fact.glm,x_no_fact,x_fact)
    x.inputs[sapply(x.inputs, is.null)] <- NULL
  }

  now <- Sys.time()
  fn <- paste0("cann_best_weights_",format(now, "%Y%m%d_%H%M%S.hdf5"))

  CBs <- callback_model_checkpoint(fn, monitor=layer$method_options$monitor, save_best_only = TRUE, save_weights_only = TRUE)

  layer$history <- CANN %>%
    keras::fit(x=x.inputs, y=y, sample_weight=sample.w, epochs = layer$method_options$epochs,
               batch_size = layer$method_options$batch_size,
               validation_split = layer$method_options$validation_split,
               callbacks = list(earlystopping,CBs),
               verbose = layer$method_options$verbose)

  load_model_weights_hdf5(CANN, fn)

  if(!layer$method_options$bias_regularization) {
  layer$fit <- CANN
  }
  else {
    # We keep track of the neural network (biased) model
    layer$fit.biased <- CANN
    # Source: Ferrario, Andrea and Noll, Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April 23, 2020), p.52
    glm.formula <- function(nb) {
      #string <- "yy ~ logpred + X1"
      string <- "yy ~ X1"
      if(nb>1) {for (i in 2:nb) {string <- paste(string," + X",i, sep="")}}
      string
    }

    glm.formula.2 <- function(cov) {
      string <- "yy ~ "
      for (i in 2:length(cov)) {
        if(i==2) string <- paste0(string,' ',cov[i])
        else string <- paste0(string,' + ',cov[i])
      }
      string
    }

    #zz        <- keras_model(inputs = CANN$input, outputs=get_layer(CANN,'output_layer_CANN')$output)
    zz        <- keras_model(inputs = CANN$input, outputs=get_layer(CANN,'output_layer_NN')$output)
    #zz        <- keras_model(inputs = CANN$input, outputs=get_layer(CANN,'last_hidden_layer_activation')$output)
    layer$zz  <- zz

    glm.pred <- layer$model.glm$fitted.values

    Zlearn    <- data.frame(zz %>% predict(x.inputs))
    names(Zlearn) <- paste0('X', 1:ncol(Zlearn))
    # We keep track of the pre-processed data for analysis purposes
    layer$Zlearn <- Zlearn

    Zlearn$yy <- y
    Zlearn$glm.pred <- glm.pred
    data$glm.pred <- glm.pred

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

    glm1 <- glm(as.formula(glm.formula(ncol(Zlearn)-2)), data=Zlearn, family=fam, weights = weights.vec)
    #glm1 <- glm(as.formula(glm.formula(ncol(Zlearn)-1)), data=Zlearn, family=fam, weights = weights.vec)
    cov <- names(glm1$coefficients[!sapply(glm1$coefficients,is.na)])
    if(length(cov)>0) {
      glm2 <- glm(as.formula(glm.formula.2(cov)), data=Zlearn, family=fam, weights = weights.vec)
      layer$fit <- glm2
    }
    else layer$fit <- glm1
  }

  if(!is.null(obj$balance.var)){
    layer$balance.correction <- sapply(data %>% group_split(data[[obj$balance.var]]),
                                       function(x) {

                                         data_baked_bc     <- bake(layer$data_recipe, new_data = x)
                                         if(ncol(data_baked_bc) == 1)
                                           data_baked_bc <- data_baked_bc %>% mutate(intercept = 1)

                                         data_baked_bc.glm <- bake(layer$data_recipe.glm, new_data = x)

                                         if(ncol(data_baked_bc.glm) == 1)
                                           data_baked_bc.glm <- data_baked_bc.glm %>% mutate(intercept = 1)

                                         def_x <- def_x(layer$method_options$use_embedding,
                                                        layer$method_options$embedding_var,
                                                        layer$method_options$embedding_var.glm,
                                                        f,
                                                        f.glm,
                                                        x,
                                                        data_baked_bc,
                                                        data_baked_bc.glm,
                                                        layer$data_recipe,
                                                        layer$data_recipe.glm,
                                                        label)

                                         if(!layer$method_options$use_embedding) {
                                           x.tmp <- def_x$x
                                           x.tmp.glm <- def_x$x.glm
                                           x.inputs.tmp <- list(x.tmp,x.tmp.glm)
                                         }
                                         else {
                                           x_fact.tmp         <- def_x$x_fact
                                           x_no_fact.tmp      <- def_x$x_no_fact
                                           x_fact.glm.tmp     <- def_x$x_fact.glm
                                           x_no_fact.glm.tmp  <- def_x$x_no_fact.glm
                                           x.inputs.tmp <- list(x_no_fact.glm.tmp,x_fact.glm.tmp,x_no_fact.tmp,x_fact.tmp)
                                           x.inputs.tmp[sapply(x.inputs.tmp, is.null)] <- NULL
                                         }

                                         if(layer$method_options$bias_regularization) {
                                           Zlearn.tmp   <- data.frame(layer$zz %>% predict(x.inputs.tmp))
                                           names(Zlearn.tmp) <- paste0('X', 1:ncol(Zlearn.tmp))
                                           Zlearn.tmp$glm.pred <- x$glm.pred
                                           sum(x[[layer$name]])/sum(predict(layer$fit, newdata = Zlearn.tmp, type = 'response'))
                                         }
                                         else {
                                           sum(x[[layer$name]])/sum(layer$fit %>% predict(x.inputs.tmp))
                                         }

                                       })
  }

  if(layer$method_options$bias_regularization)
    pred <- layer$fit$fitted.values
  else
    pred <- layer$fit %>% predict(x.inputs)

  if(layer$method_options$distribution == 'gaussian') {
    layer$sigma <- sd(pred - y)
  }

  if(layer$method_options$distribution == 'gamma') {
    shape <- hirem_gamma_shape(observed = y, fitted = pred, weight = weights.vec)
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
#' @importFrom purrr map_chr
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

