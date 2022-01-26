

#' @importFrom xgboost xgb.DMatrix xgb.cv xgb.train
#' @importFrom recipes recipe step_dummy bake prep all_nominal all_numeric all_outcomes
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
  #f <- as.formula(paste0(formula,'-1'))
  f <- as.formula(formula)
  label <- as.character(terms(f)[[2]])

  if(!is.null(layer$transformation)) {
    data[,label] <- layer$transformation$transform(data[,label])
  }

  data_recipe <- recipe(as.formula(formula), data=data)
  data_recipe <- data_recipe %>% step_dummy(all_nominal(), one_hot = TRUE)
  data_recipe <- data_recipe %>% prep()
  layer$data_recipe <- data_recipe

  data_baked <- bake(data_recipe, new_data = data)
  x <- select(data_baked,-as.name(label)) %>% as.matrix()
  data.xgb <- xgb.DMatrix(data = x, info = list('label' = as.matrix(data[,label])))

  if(!is.null(obj$weights)) {
    weights.vec <- obj$weights[data[[obj$weight.var]]]
    weights.vec.n <- weights.vec*length(weights.vec)/sum(weights.vec)
    xgboost::setinfo(data.xgb,'weight',weights.vec.n)
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
        colsample_bynode = 1,
        gamma = c(0),
        lambda = c(0,.1),
        alpha = c(0,.1)
      )
    }

    if(layer$method_options$random_trials > 0) {
      i <- sample(1:dim(hyper_grid)[1],layer$method_options$random_trials)
      hyper_grid <- hyper_grid[i,]
    }

    best_eval_metric = 10^6

    mean_scores <- c()
    best_nrounds <- c()

    for(i in seq_len(nrow(hyper_grid))) {

      params = list(
        booster = layer$method_options$booster
        , tree_method = layer$method_options$tree_method
        , grow_policy = layer$method_options$grow_policy
        , eval_metric = layer$method_options$eval_metric
        , eta = hyper_grid$eta[i]
        , max_depth = hyper_grid$max_depth[i]
        , min_child_weight = hyper_grid$min_child_weight[i]
        , subsample = hyper_grid$subsample[i]
        , colsample_bynode = hyper_grid$colsample_bynode[i]
        , gamma = hyper_grid$gamma[i]
        , lambda = hyper_grid$lambda[i]
        , alpha = hyper_grid$alpha[i]
        , max_delta_step = layer$method_options$max_delta_step
        , objective = layer$method_options$objective
        , eval_metric = layer$method_options$eval_metric
        , nthread = layer$method_options$nthread
        #, scale_pos_weight = layer$method_options$scale_pos_weight
      )

      xval <- xgb.cv(
        data = data.xgb,
        nrounds = layer$method_options$nrounds,
        early_stopping_rounds = layer$method_options$early_stopping_rounds,
        nfold = layer$method_options$nfolds,
        stratified = layer$method_options$stratified,
        verbose = layer$method_options$verbose,
        params = params)


      min_eval_metric <- min(xval$evaluation_log[,4])
      min_eval_index <- as.numeric(which.min(as.matrix(xval$evaluation_log[,4])))

      mean_scores[i] <- min_eval_metric
      best_nrounds[i] <- min_eval_index

      if (min_eval_metric < best_eval_metric) {
        best_eval_metric = min_eval_metric
        best_eval_index = min_eval_index
        best_param = params
        best_xval = xval
        best_ntreelimit = xval$best_ntreelimit
      }
    }

    nrounds <- best_eval_index
    layer$best_score <- best_eval_metric
    params <- best_param

    layer$hyper_grid <- hyper_grid
    layer$hyper_grid$mean_scores <- mean_scores
    layer$hyper_grid$best_nrounds <- best_nrounds
    layer$hyper_grid <- layer$hyper_grid %>% arrange(mean_scores)
    layer$best_xval <- best_xval
    layer$best_ntreelimit <- best_ntreelimit

  }
  else if(layer$method_options$bayesOpt) {

    # Folds <- list()
    # names <- c()
    # for(i in 1:layer$method_options$nfolds) {
    #   Folds[[i]] <- as.integer(seq(i,nrow(data.xgb),by = layer$method_options$nfolds))
    #   names[i] <- paste0('Fold',i)
    # }
    # names(Folds) <- names

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
        #, scale_pos_weight = scale_pos_weight
      )

      xgbcv <- xgb.cv(
        params = Pars
        , data = data.xgb
        , nround = layer$method_options$nrounds
        , nfold = layer$method_options$nfolds
        , early_stopping_rounds = layer$method_options$early_stopping_rounds
        , stratified = layer$method_options$stratified
        , verbose = layer$method_options$verbose
        #, maximize = !layer$method_options$bayesOpt.min
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
    else {
      bounds <- layer$method_options$bayesOpt_bounds
    }

    bounds_names <- as.vector(names(bounds))

    tNoPar <- system.time(
      optObj <- bayesOpt(
        FUN = scoringFunction
        , bounds = bounds
        #, initPoints = max(length(bounds)+1,3)
        , initPoints = layer$method_options$bayesOpt_initPoints
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
      #scale_pos_weight = ifelse('scale_pos_weight' %in% bounds_names,getBestPars(optObj)$scale_pos_weight,layer$method_options$scale_pos_weight)
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
    #early_stopping_rounds = layer$method_options$early_stopping_rounds,
    verbose = layer$method_options$verbose,
    params = params
  )

  layer$best_params <- params
  #layer$nrounds <- nrounds

  #print('best params')
  #print(params)

  if(!is.null(obj$balance.var)){
    layer$balance.correction <- sapply(data %>% split(data[[obj$balance.var]]),
                                       function(x) {
                                         data_baked <- bake(data_recipe, new_data = x)
                                         nd <- select(data_baked,-as.name(label)) %>% as.matrix()
                                         newdata <- xgb.DMatrix(data = nd, info = list('label' = as.matrix(x[,label])))
                                         sum(x[[layer$name]])/sum(predict(layer$fit, ntreelimit = layer$fit$niter, newdata = newdata,type = 'response'))
                                         }
                                       )
  }

  if(layer$method_options$objective == 'reg:squarederror') {
    layer$sigma <- sd(predict(layer$fit, ntreelimit = layer$fit$niter, newdata = data.xgb, type = "response") - as.matrix(data[,label]))
  }

  if(layer$method_options$objective == 'reg:gamma') {

    if(is.null(obj$weights)) weights.vec.normalized <- NULL
    else weights.vec.normalized <- weights.vec*length(weights.vec)/sum(weights.vec)

    shape <- hirem_gamma_shape(observed = as.matrix(data[,label]),
                               fitted = predict(layer$fit, ntreelimit = layer$fit$niter, newdata = data.xgb, type = "response"),
                               weight = weights.vec.normalized)
    layer$shape <- shape$shape
    layer$shape.se <- shape$se
    layer$weights.vec.normalized <- weights.vec.normalized
  }

  return(layer)
}

