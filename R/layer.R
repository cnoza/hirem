#' @export
hirem_layer <- function(obj, name, method, method_class, method_options, filter = NULL, transformation = NULL) {
  layer <- c()
  layer$name <- name
  layer$method <- method

  layer$method_options <- method_options

  if(is.null(filter)) {
    layer$filter <- function(x){rep(TRUE, nrow(x))}
  } else {
    layer$filter <- filter
  }

  layer$transformation <- NULL;
  if(class(transformation) == 'hirem_transformation') {
    layer$transformation <- transformation;
  } else if(!is.null(transformation)) {
    stop(paste('expected an object of class hirem_transformation, but received', print(transformation)))
  }

  class(layer) <- c(method_class, 'hirem_layer');

  obj$layers <- append(obj$layers, list(layer))
  names(obj$layers)[length(obj$layers)] <- name

  return(obj)
}

#' Layer estimated using a generalized linear model
#'
#' Adds a new layer to the hierarchical reserving model. This layer will be estimated using the glm framework.
#'
#' @param obj The hierarchical reserving model
#' @param name Character, name of the layer. This name should match the variable name in the data set
#' @param family family argument passed to the \code{\link[stats]{glm}} function
#' @param filter Function with \itemize{
#'   \item input: Data set with same structure as the data passed to \code{\link{hirem}}
#'   \item output: TRUE/FALSE vector with same length as the number of rows in the input data set.\cr
#'         FALSE indicates that this layer is zero for the current record.
#'  }
#' @param transformation Object of class \code{\link{hirem_transformation}} specifying the transformation
#' applied before modelling this layer.
#' @export
layer_glm <- function(obj, name, family, filter = NULL, transformation = NULL) {
  hirem_layer(obj, name, 'glm', 'layer_glm', family, filter, transformation)
}

#' Layer estimated using a gradient boosting model
#'
#' Adds a new layer to the hierarchical reserving model. This layer will be estimated using the \code{\link[gbm]{gbm}} package.
#'
#' @param obj The hierarchical reserving model
#' @param name Character, name of the layer. This name should match the variable name in the data set
#' @param distribution distribution argument passed to \code{\link[gbm]{gbm}}
#' @param n.trees n.trees argument passed to \code{\link[gbm]{gbm}}
#' @param interaction.depth interaction.depth argument passed to \code{\link[gbm]{gbm}}
#' @param n.minobsinnode n.minobsinnode argument passed to \code{\link[gbm]{gbm}}
#' @param shrinkage shrinkage argument passed to \code{\link[gbm]{gbm}}
#' @param bag.fraction bag.fraction argument passed to \code{\link[gbm]{gbm}}
#' @param select_trees Character string specifying the method for selecting the optimal number of trees after fitting the gbm \itemize{
#'    \item "fixed": Use the number of trees specified in n.trees
#'    \item "perf": Update the number of trees using \code{\link[gbm]{gbm.perf}}
#' }
#' @param filter Function with \itemize{
#'   \item input: Data set with same structure as the data passed to \code{\link{hirem}}
#'   \item output: TRUE/FALSE vector with same length as the number of rows in the input data set.\cr
#'         FALSE indicates that this layer is zero for the current record.
#'  }
#' @param transformation Object of class \code{\link{hirem_transformation}} specifying the transformation
#' applied before modelling this layer.
#' @export
layer_gbm <- function(obj, name, distribution, n.trees = 500, interaction.depth = 2, n.minobsinnode = 10,
                      shrinkage = 0.1, bag.fraction = 0.5, cv.folds = 0, select_trees = 'perf', filter = NULL, transformation = NULL) {

  options <- c()
  options$shrinkage <- shrinkage
  options$n.minobsinnode <- n.minobsinnode
  options$interaction.depth <- interaction.depth
  options$n.trees <- n.trees
  options$select_trees <- select_trees
  options$bag.fraction <- bag.fraction
  options$distribution <- distribution
  options$cv <- cv.folds

  hirem_layer(obj, name, 'gbm', 'layer_gbm', options, filter, transformation)
}

#' Layer estimated using an extreme gradient boosting model
#'
#' Adds a new layer to the hierarchical reserving model. This layer will be estimated using the \code{\link[xgboost]{xgboost}} package.
#'
#' @param obj The hierarchical reserving model
#' @param name Character, name of the layer. This name should match the variable name in the data set
#' @param nrounds Max number of boosting iterations, passed to \code{\link[xgboost]{xgboost}}. Default is 1500.
#' @param early_stopping_rounds Passed to \code{\link[xgboost]{xgboost}}. If NULL, the early stopping function is not triggered. If set to an integer k, training with a validation set will stop if the performance doesn't improve for k rounds. Default is 20.
#' @param verbose If 0, \code{\link[xgboost]{xgboost}} will stay silent. If 1, it will print information about performance. Default is 0.
#' @param booster Passed to \code{\link[xgboost]{xgboost}}. Which booster to use, can be gbtree or gblinear. Default is gbtree.
#' @param objective Specify the learning task and the corresponding learning objective, passed to \code{\link[xgboost]{xgboost}}. Default options is \code{binary:logistic}.
#' @param eval_metric Evaluation metrics for validation data. Default is 'poisson-nloglik'.
#' @param eta The learning rate passed to \code{\link[xgboost]{xgboost}}. Default is 0.01
#' @param nthread Number of parallel threads used to run \code{\link[xgboost]{xgboost}}. Default is 1.
#' @param subsample Subsample ratio of the training instance. Default is 0.75. Setting it to 0.75 means that \code{\link[xgboost]{xgboost}} randomly collected 75 percent of the data instances to grow trees and this will prevent overfitting.
#' @param colsample_bynode Subsample ratio of columns for each node (split). Passed to \code{\link[xgboost]{xgboost}}
#' @param max_depth Maximum depth of a tree, passed to \code{\link[xgboost]{xgboost}}. Default is 3
#' @param min_child_weight Minimum sum of instance weight (hessian) needed in a child, passed to \code{\link[xgboost]{xgboost}}. Default is 1000.
#' @param gamma Minimum loss reduction required to make a further partition on a leaf node of the tree, passed to \code{\link[xgboost]{xgboost}}. Default is 0.
#' @param lambda L2 regularization term on weights, passed to \code{\link[xgboost]{xgboost}}. Default is 0.01.
#' @param alpha L1 regularization term on weights, passed to \code{\link[xgboost]{xgboost}}. Default is 0.01.
#' @param filter Function with \itemize{
#'   \item input: Data set with same structure as the data passed to \code{\link{hirem}}
#'   \item output: TRUE/FALSE vector with same length as the number of rows in the input data set.\cr
#'         FALSE indicates that this layer is zero for the current record.
#'  }
#' @param transformation Object of class \code{\link{hirem_transformation}} specifying the transformation
#' applied before modelling this layer.
#' @export
layer_xgb <- function(obj, name, nrounds = 1500, early_stopping_rounds = 20, verbose = F, booster = 'gbtree', objective = 'count:poisson',
                      eval_metric = 'poisson-nloglik', eta = 0.01, nthread = 1, subsample = .75, colsample_bynode = .5, max_depth = 3,
                      min_child_weight = 1000, gamma = 0, lambda = .01, alpha = .01, filter = NULL, transformation = NULL) {

  options <- c()
  options$nrounds <- nrounds
  options$early_stopping_rounds <- early_stopping_rounds
  options$verbose <- verbose
  options$booster <- booster
  options$objective <- objective
  options$eval_metric <- eval_metric
  options$eta <- eta
  options$nthread <- nthread
  options$subsample <- subsample
  options$colsample_bynode <- colsample_bynode
  options$max_depth <- max_depth
  options$min_child_weight <- min_child_weight
  options$gamma <- gamma
  options$lambda <- lambda
  options$alpha <- alpha

  hirem_layer(obj, name, 'xgb', 'layer_xgb', options, filter, transformation)
}

#' @export
print.hirem_layer <- function(obj, ...) {
  if(!is.null(obj$transformation)) {
    cat(obj$transformation$name, '(', obj$name, ')', sep = '')
  } else {
    cat(obj$name, sep='')
  }

  cat(': ', obj$method, ' ', sep='')

  if(class(obj)[1] == 'layer_glm' & class(obj$method_options) == 'family') {
    cat(obj$method_options$family, ' (', obj$method_options$link,') ', sep='')
  }

  cat('\n')
  if(!is.null(obj$formula)) {
    print(obj$formula);
  }
}

#' @export
hirem_get_layer_pos <- function(obj, layer) {

  if(length(obj$layers) == 0) {
    stop('There are no layers defined for the supplied hierarchical reserving model.')
  }

  names <- map_chr(obj$layers, 'name')
  if(!(layer %in% names)) {
    stop(paste('Layer ', layer, ' not found.', sep=''))
  }

  return(match(layer, names))
}
