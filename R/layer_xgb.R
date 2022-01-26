
#' Layer estimated using an extreme gradient boosting model
#'
#' Adds a new layer to the hierarchical reserving model. This layer will be estimated using the \code{\link[xgboost]{xgboost}} package.
#'
#' @param obj The hierarchical reserving model
#' @param name Character, name of the layer. This name should match the variable name in the data set
#' @param nrounds Max number of boosting iterations, passed to \code{xgboost}. Default is 100.
#' @param early_stopping_rounds Passed to \code{xgboost}. If NULL, the early stopping function is not triggered. If set to an integer k, training with a validation set will stop if the performance doesn't improve for k rounds. Default is 20.
#' @param verbose If 0, \code{xgboost} will stay silent. If 1, it will print information about performance. Default is 0.
#' @param booster Passed to \code{xgboost}. Which booster to use, can be gbtree or gblinear. Default is gbtree.
#' @param objective Specify the learning task and the corresponding learning objective, passed to \code{xgboost}.
#' @param stratified If TRUE, use stratified sampling in cross-validation. Default is FALSE.
#' @param grow_policy Passed to \code{xgboost}. Default is depthwise.
#' @param eval_metric Evaluation metrics for validation data. Default is 'rmse'.
#' @param eta The learning rate passed to \code{xgboost}. Default is 0.05.
#' @param nthread Number of parallel threads used to run \code{xgboost}. Default is 1.
#' @param subsample Subsample ratio of the training instance. Default is 0.8. Setting it to 0.8 means that \code{xgboost} randomly collected 80 percent of the data instances to grow trees and this will prevent overfitting.
#' @param colsample_bynode Subsample ratio of columns for each node (split). Passed to \code{xgboost}
#' @param max_depth Maximum depth of a tree, passed to \code{xgboost}. Default is 2.
#' @param min_child_weight Minimum sum of instance weight (hessian) needed in a child, passed to \code{xgboost}. Default is 10.
#' @param gamma Minimum loss reduction required to make a further partition on a leaf node of the tree, passed to \code{xgboost}. Default is 0.
#' @param lambda L2 regularization term on weights, passed to \code{xgboost}. Default is 0.01.
#' @param alpha L1 regularization term on weights, passed to \code{xgboost}. Default is 0.01.
#' @param gridsearch_cv If TRUE, hyperparameters are tuned following a gridsearch and cross-validation strategy.
#' @param bayesOpt If TRUE, hyperparameters are tuned following a Bayesian optimization strategy.
#' @param nfolds If \code{gridsearch_cv} is TRUE, \code{nfolds} can be used to set the number of folds to consider in the cross-validation. Default is 5.
#' @param hyper_grid If \code{nfolds} is not null, the set of tuning parameters can be given in \code{hyper_grid}. If NULL, a default set of parameters is used.
#' @param filter Function with \itemize{
#'   \item input: Data set with same structure as the data passed to \code{hirem}
#'   \item output: TRUE/FALSE vector with same length as the number of rows in the input data set.\cr
#'         FALSE indicates that this layer is zero for the current record.
#'  }
#' @param transformation Object of class \code{hirem_transformation} specifying the transformation
#' applied before modelling this layer.
#' @export
layer_xgb <- function(obj, name, nrounds = 1000, early_stopping_rounds = 20, verbose = 0, booster = 'gbtree', objective, stratified = T, grow_policy = 'depthwise',
                      eval_metric = 'rmse', eta = 0.05, nthread = 1, subsample = 1, colsample_bynode = 1, max_depth = 6, max_delta_step = 0, scale_pos_weight = 1,
                      min_child_weight = 100, gamma = 0, lambda = 1, alpha = 0, hyper_grid = NULL, gridsearch_cv = FALSE, nfolds = 5, tree_method = 'auto',
                      bayesOpt = FALSE, bayesOpt_min = FALSE, bayesOpt_iters_n = 3, bayesOpt_bounds = NULL, bayesOpt_initPoints = 4, random_trials = 0,
                      filter = NULL, transformation = NULL) {

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
  options$max_delta_step <- max_delta_step
  options$min_child_weight <- min_child_weight
  options$gamma <- gamma
  options$lambda <- lambda
  options$alpha <- alpha
  options$nfolds <- nfolds
  options$stratified <- stratified
  options$tree_method <- tree_method
  options$grow_policy <- grow_policy
  options$hyper_grid <- hyper_grid
  options$gridsearch_cv <- gridsearch_cv
  options$bayesOpt <- bayesOpt
  options$bayesOpt.min <- bayesOpt_min
  options$bayesOpt_iters_n <- bayesOpt_iters_n
  options$bayesOpt_bounds <- bayesOpt_bounds
  options$bayesOpt_initPoints <- bayesOpt_initPoints
  options$scale_pos_weight <- scale_pos_weight
  options$random_trials <- random_trials

  if(options$gridsearch_cv & options$bayesOpt)
    stop('Those options, if TRUE, are mutually exclusive.')

  hirem_layer(obj, name, 'xgb', 'layer_xgb', options, filter, transformation)
}

