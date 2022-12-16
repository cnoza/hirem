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

    if(length(layer$balance.correction) != length(levels(data.filter$dev.year.fact))) {
      ind <- !(levels(data.filter$dev.year.fact) %in% names(layer$balance.correction))
      layer$balance.correction[as.character(which(ind))]=1
      layer$balance.correction = layer$balance.correction[levels(data.filter$dev.year.fact)]
    }

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
  layer$weights.vec <- weights.vec

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

    if(length(layer$balance.correction) != length(levels(data$dev.year.fact))) {
      ind <- !(levels(data$dev.year.fact) %in% names(layer$balance.correction))
      layer$balance.correction[as.character(which(ind))]=1
      layer$balance.correction = layer$balance.correction[levels(data$dev.year.fact)]
    }

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

