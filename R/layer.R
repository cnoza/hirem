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
#' @param family family argument passed to the \code{glm} function
#' @param filter Function with \itemize{
#'   \item input: Data set with same structure as the data passed to \code{hirem}
#'   \item output: TRUE/FALSE vector with same length as the number of rows in the input data set.\cr
#'         FALSE indicates that this layer is zero for the current record.
#'  }
#' @param transformation Object of class \code{hirem_transformation} specifying the transformation
#' applied before modelling this layer.
#' @export
layer_glm <- function(obj, name, family, filter = NULL, transformation = NULL) {
  hirem_layer(obj, name, 'glm', 'layer_glm', family, filter, transformation)
}

#' Layer estimated using a gradient boosting model
#'
#' Adds a new layer to the hierarchical reserving model. This layer will be estimated using the \code{gbm} package.
#'
#' @param obj The hierarchical reserving model
#' @param name Character, name of the layer. This name should match the variable name in the data set
#' @param distribution distribution argument passed to \code{gbm}
#' @param n.trees n.trees argument passed to \code{gbm}
#' @param interaction.depth interaction.depth argument passed to \code{gbm}
#' @param n.minobsinnode n.minobsinnode argument passed to \code{gbm}
#' @param shrinkage shrinkage argument passed to \code{gbm}
#' @param bag.fraction bag.fraction argument passed to \code{gbm}
#' @param select_trees Character string specifying the method for selecting the optimal number of trees after fitting the gbm \itemize{
#'    \item "fixed": Use the number of trees specified in n.trees
#'    \item "perf": Update the number of trees using \code{gbm.perf}
#' }
#' @param filter Function with \itemize{
#'   \item input: Data set with same structure as the data passed to \code{hirem}
#'   \item output: TRUE/FALSE vector with same length as the number of rows in the input data set.\cr
#'         FALSE indicates that this layer is zero for the current record.
#'  }
#' @param transformation Object of class \code{hirem_transformation} specifying the transformation
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
                      bayesOpt = FALSE, bayesOpt_min = FALSE, bayesOpt_iters_n = 3, bayesOpt_bounds = NULL, bayesOpt_initPoints = 4,
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
  options$select_trees <- select_trees

  if(options$gridsearch_cv & options$bayesOpt)
    stop('Those options, if TRUE, are mutually exclusive.')

  hirem_layer(obj, name, 'xgb', 'layer_xgb', options, filter, transformation)
}

#' Layer estimated using a multi-layer perceptron model (h2o)
#'
#' Adds a new layer to the hierarchical reserving model. This layer will be estimated using the \code{deeplearning} function of the \code{\link[h2o]{h2o}} package.
#'
#' @param obj The hierarchical reserving model
#' @param name Character, name of the layer. This name should match the variable name in the data set
#' @param distribution The distribution used for the simulation. Default is gaussian.
#' @param hidden The hidden layer architecture passed to \code{h2o}. Default is c(10,10).
#' @param epochs The epochs argument passed to \code{h2o}. Default is 1000
#' @param train_samples_per_iteration The train_samples_per_iteration argument passed to \code{h2o}. Default is -1
#' @param reproducible The reproducible argument passed to \code{h2o}. Default is True.
#' @param activation The activation function (same for all layers) passed to \code{h2o}. Default is "Tanh".
#' @param single_node_mode The single_node_mode argument passed to \code{h2o}. Default is FALSE,
#' @param balance_classes The balance_classes argument passed to \code{h2o}. Default is FALSE,
#' @param force_load_balance The force_load_balance argument passed to \code{h2o}. Default is FALSE,
#' @param seed The seed argument passed to \code{h2o}.
#' @param tweedie_power The tweedie_power argument passed to \code{h2o}. Only used if distribution is "tweedie".
#' @param score_training_samples The score_training_samples argument passed to \code{h2o}. Default is 0,
#' @param score_validation_samples The score_validation_samples argument passed to \code{h2o}. Default is 0,
#' @param stopping_rounds The stopping_rounds argument passed to \code{h2o}. Default is 0
#' @param input_dropout_ratio The input_dropout_ratio argument passed to \code{h2o}. Default is 0.1
#' @param hidden_dropout_ratios The input_dropout_ratio argument passed to \code{h2o}. Default is 0.5
#' @param stopping_rounds The stopping_rounds argument passed to \code{h2o}. Default is 0
#' @param filter Function with \itemize{
#'   \item input: Data set with same structure as the data passed to \code{hirem}
#'   \item output: TRUE/FALSE vector with same length as the number of rows in the input data set.\cr
#'         FALSE indicates that this layer is zero for the current record.
#'  }
#' @param transformation Object of class \code{hirem_transformation} specifying the transformation
#' applied before modelling this layer.
#' @export
layer_dnn_h2o <- function(obj, name, distribution = "gaussian", hidden = c(10,10), epochs = 1000, train_samples_per_iteration = -1,
                     reproducible = T, activation = "Tanh", nfolds = NULL,
                     single_node_mode = FALSE,
                     balance_classes = FALSE,
                     force_load_balance = FALSE,
                     seed = NULL,
                     tweedie_power = 1.5,
                     score_training_samples = 0,
                     score_validation_samples = 0,
                     input_dropout_ratio = 0.1,
                     hidden_dropout_ratios = 0.5,
                     stopping_rounds = 0, filter = NULL, transformation = NULL) {

  options <- c()
  options$distribution <- distribution
  options$hidden <- hidden
  options$epochs <- epochs
  options$nfolds <- nfolds
  options$train_samples_per_iteration <- train_samples_per_iteration
  options$reproducible <- reproducible
  options$activation <- activation
  options$single_node_mode <- single_node_mode
  options$balance_classes <- balance_classes
  options$force_load_balance <- force_load_balance
  options$seed <- seed
  options$tweedie_power <- tweedie_power
  options$score_training_samples <- score_training_samples
  options$score_validation_samples <- score_validation_samples
  options$stopping_rounds <- stopping_rounds
  options$input_dropout_ratio <- input_dropout_ratio
  options$hidden_dropout_ratios <- hidden_dropout_ratios

  hirem_layer(obj, name, 'dnn_h2o', 'layer_dnn_h2o', options, filter, transformation)
}

#' Layer estimated using a multi-layer perceptron model with Keras
#'
#' Adds a new layer to the hierarchical reserving model. This layer will be estimated using the \code{\link[keras]{keras}} package.
#'
#' @param obj The hierarchical reserving model
#' @param name Character, name of the layer. This name should match the variable name in the data set
#' @param distribution The distribution used for the simulation. Default is gaussian.
#' @param use_bias Argument passed to \code{keras} in the definition of the MLP architecture.
#' @param ae.hidden The hidden layer architecture of a (stacked) autoencoder to be fitted on the input data.
#' If used, then the encoding part is used to compress the input data, before fitting the MLP model.
#' @param ae.activation.hidden The activation functions to be used in the hidden layer architecture of the (stacked) autoencoder.
#' If NULL, the linear activation function is used in all hidden layers.
#' @param hidden The hidden layer architecture passed to \code{keras}.
#' @param dropout.hidden The dropout ratios for each hidden layer passed to \code{keras}. Default is 0.
#' @param step_log If TRUE, the logarithmic transformation is applied on the response variable through the use of a \code{recipe}.
#' @param step_normalize If TRUE, the \code{step_normalize} function is used to preprocess the input data through the use of a \code{recipe}.
#' @param activation.hidden The activation function for each hidden layer passed to \code{keras}. Default is Relu
#' @param activation.output The activation function for the output layer passed to \code{keras}. Default is Linear
#' @param batch_normalization If TRUE (default), apply the batch normalization between each hidden layer.
#' @param loss The loss function argument passed to \code{keras}. Default is 'mse'.
#' @param monitor The monitor argument passed to \code{keras}. Default is 'loss'.
#' @param patience The patience argument passed to \code{keras}. Default is 20.
#' @param scale If TRUE (default), the data used for training is scaled.
#' @param optimizer The optimizer argument passed to \code{keras}. Default is 'nadam'.
#' @param epochs The epochs argument passed to \code{keras}. Default is 20.
#' @param nfolds The number of folds for cross-validation. Default is 5. Set it to 1 to deactivate cross-validation.
#' @param batch_size The batch_size argument passed to \code{keras}. Default is 1000.
#' @param validation_split The validation_split argument passed to \code{keras}. Default is .2
#' @param verbose The verbose argument passed to the \code{fit} function of \code{keras}. Default is 1.
#' @param metrics The metrics argument passed to \code{keras}.
#' @param family_for_init If not NULL, an homogenous GLM is estimated and the resulting coefficient estimate is used to initialize the bias weight in the output layer, which may improve convergence.
#' See Ferrario, Andrea and Ferrario, Andrea and Noll, Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April 23, 2020). Available at SSRN: https://ssrn.com/abstract=3226852 or http://dx.doi.org/10.2139/ssrn.3226852
#' @param filter Function with \itemize{
#'   \item input: Data set with same structure as the data passed to \code{hirem}
#'   \item output: TRUE/FALSE vector with same length as the number of rows in the input data set.\cr
#'         FALSE indicates that this layer is zero for the current record.
#'  }
#' @param transformation Object of class \code{hirem_transformation} specifying the transformation
#' applied before modelling this layer.
#' @export
layer_dnn <- function(obj, name, distribution = 'gaussian', use_bias = TRUE, ae.hidden = NULL, ae.activation.hidden = NULL,
                            hidden = NULL, dropout.hidden = NULL, step_log = FALSE, step_normalize = FALSE, verbose = 0,
                            activation.hidden = NULL, activation.output = 'linear', batch_normalization = FALSE, use_embedding = FALSE, output_dim = 1, embedding_var = c(),
                            loss = 'mse', optimizer = 'nadam', epochs = 20, batch_size = 1000, validation_split = .2, metrics = NULL,
                            monitor = "loss", patience = 20, family_for_init = NULL, nfolds = 5, bias_regularization = TRUE, shuffle = TRUE,
                            bayesOpt = FALSE, bayesOpt_min = FALSE, bayesOpt_iters_n = 3, bayesOpt_bounds = NULL, bayesOpt_initPoints = 4, bayesOpt_step = 1,
                            filter = NULL, transformation = NULL, gridsearch_cv = F, gridsearch_cv.min = T, hyper_grid = NULL, one_hot = TRUE) {

  options <- c()
  options$step_log <- step_log
  options$step_normalize <- step_normalize
  options$distribution <- distribution
  options$hidden <- hidden
  options$dropout.hidden <- dropout.hidden
  options$activation.hidden <- activation.hidden
  options$use_bias <- use_bias
  options$activation.output <- activation.output
  options$loss <- loss
  options$optimizer <- optimizer
  options$epochs <- epochs
  options$batch_size <- batch_size
  options$metrics <- metrics
  options$validation_split <- validation_split
  options$family_for_init <- family_for_init
  options$monitor <- monitor
  options$patience <- patience
  options$batch_normalization <- batch_normalization
  options$ae.hidden <- ae.hidden
  options$ae.activation.hidden <- ae.activation.hidden
  options$verbose <- verbose
  options$bayesOpt <- bayesOpt
  options$bayesOpt.min <- bayesOpt_min
  options$bayesOpt_iters_n <- bayesOpt_iters_n
  options$bayesOpt_bounds <- bayesOpt_bounds
  options$bayesOpt_initPoints <- bayesOpt_initPoints
  options$nfolds <- nfolds
  options$bias_regularization <- bias_regularization
  options$use_embedding <- use_embedding
  options$output_dim <- output_dim
  options$embedding_var <- embedding_var
  options$bayesOpt_step <- bayesOpt_step
  options$gridsearch_cv <- gridsearch_cv
  options$gridsearch_cv.min <- gridsearch_cv.min
  options$hyper_grid <- hyper_grid
  options$one_hot <- one_hot
  options$shuffle <- shuffle

  if(is.null(options$ae.hidden)) {
    if(!is.null(options$ae.activation.hidden)) stop('If ae.hidden is NULL, so should be ae.activation.hidden.')
  }
  else {
    if(is.null(options$ae.activation.hidden)) {
      print('ae.activation.hidden is NULL, initialized with \'linear\' by default.')
      options$ae.activation.hidden <- rep('linear',length(options$ae.hidden))
    }
    else {
      if(length(options$ae.hidden) != length(options$ae.activation.hidden))
        stop('The length of ae.hidden and ae.activation.hidden should match.')
    }
    if(options$use_embedding)
      stop('If ae.hidden is not NULL, embedding layers cannot be used for factor variables.')
  }

  if(is.null(options$hidden)) {
    if(!is.null(dropout.hidden)) stop('If hidden is NULL, so should be dropout.hidden.')
    if(!is.null(activation.hidden)) stop('If hidden is NULL, so should be activation.hidden.')
  }
  else {
    if(!is.null(options$dropout.hidden)) {
      if(length(options$hidden) != length(options$dropout.hidden)) {
        stop('The length of hidden and dropout.hidden should match.')
      }
    }
    if(is.null(options$activation.hidden)) options$activation.hidden <- rep('relu',length(options$hidden))
    if(length(options$hidden) != length(options$activation.hidden)) {
      stop('The length of hidden and activation.hidden should match.')
    }
  }

  # if(is.null(options$hidden)) {
  #   # If no hidden layer, no need for bias regularization
  #   options$bias_regularization <- FALSE
  # }
  # else {
  #   if(options$distribution == 'gamma') { # Link = log -> activation = exponential
  #     if(options$activation.output == 'exponential')
  #       options$bias_regularization <- TRUE
  #     else
  #       stop('For the bias regularization to work, the output layer activation function should be exponential.')
  #   }
  #   else if(options$distribution == 'bernoulli') { # Default link = logit -> activation = sigmoid
  #     if(options$activation.output == 'sigmoid')
  #       options$bias_regularization <- TRUE
  #     else
  #       stop('For the bias regularization to work, the output layer activation function should be sigmoid')
  #   }
  #   else if(options$distribution == 'poisson') { # Default link = log -> activation = exponential
  #     if(options$activation.output == 'exponential')
  #       options$bias_regularization <- TRUE
  #     else
  #       stop('For the bias regularization to work, the output layer activation function should be exponential')
  #   }
  #   else
  #     stop('Bias regularization is not supported (yet) for this choice of distribution.')
  # }

  hirem_layer(obj, name, 'dnn', 'layer_dnn', options, filter, transformation)

}

#' Layer estimated using a Combined Actuarial Neural Network (CANN)
#'
#' Adds a new layer to the hierarchical reserving model. This layer will be estimated using a CANN architecture,
#' see Schelldorfer, J., & Wuthrich, M. (2019). Nesting Classical Actuarial Models into Neural Networks. Applied Computing eJournal.
#'
#' @param obj The hierarchical reserving model
#' @param name Character, name of the layer. This name should match the variable name in the data set
#' @param distribution The distribution used for the simulation. Default is gaussian,
#' @param family_for_glm The family used to estimate the GLM model. The coefficient estimates are then used to initialize the weights of the associated GLM neural network.
#' Default is Gamma(link = log).
#' @param hidden The hidden layer architecture of the Neural Network, passed to \code{keras}. Default is c(30,20,10).
#' @param dropout.hidden The dropout ratios for each hidden layer of the Neural Network, passed to \code{keras}. Default is 0.
#' @param step_log If TRUE, the logarithmic transformation is applied on the response variable through the use of a \code{recipe}.
#' @param step_normalize If TRUE, the \code{step_normalize} function is used to preprocess the input data through the use of a \code{recipe}.
#' @param activation.hidden The activation function for each hidden layer of the Neural Network, passed to \code{keras}. Default is tanh.
#' @param activation.output The activation function for the output layer of the Neural Network, passed to \code{keras}. Default is linear.
#' @param activation.output.cann The activation function for the output layer of the CANN, passed to \code{keras}. Default is linear.
#' @param fixed.cann If TRUE (default), the weights of the CANN's output layer are fixed and non trainable.
#' @param monitor The monitor argument passed to \code{keras}. Default is 'loss'.
#' @param patience The patience argument passed to \code{keras}. Default is 20.
#' @param loss The loss function argument passed to \code{keras}. Default is 'mse'.
#' @param optimizer The optimizer argument passed to \code{keras}. Default is 'nadam'.
#' @param epochs The epochs argument passed to \code{keras}. Default is 20.
#' @param batch_size The batch_size argument passed to \code{keras}. Default is 1000.
#' @param validation_split The validation_split argument passed to \code{keras}. Default is .2
#' @param metrics The metrics argument passed to \code{keras}.
#' @param filter Function with \itemize{
#'   \item input: Data set with same structure as the data passed to \code{hirem}
#'   \item output: TRUE/FALSE vector with same length as the number of rows in the input data set.\cr
#'         FALSE indicates that this layer is zero for the current record.
#'  }
#' @param transformation Object of class \code{hirem_transformation} specifying the transformation
#' applied before modelling this layer.
#' @export
#'
layer_cann <- function(obj, name, distribution = 'gaussian', family_for_glm = Gamma(link = log), use_bias = TRUE, formula.glm = NULL,
                       hidden = NULL, dropout.hidden = NULL, step_log = FALSE, step_normalize = FALSE, use_embedding = FALSE, bias_regularization = NULL,
                       activation.hidden = NULL, activation.output = 'linear', activation.output.cann = 'linear', embedding_var = c(), embedding_var.glm = c(),
                       fixed.cann = TRUE, batch_normalization = FALSE, monitor = 'loss', patience = 20, verbose = 0, shuffle = TRUE,
                       loss = 'mse', optimizer = 'nadam', epochs = 20, batch_size = 1000, validation_split = .2, metrics = NULL,
                       nfolds = 5, bayesOpt = FALSE, bayesOpt_min = FALSE, bayesOpt_iters_n = 3, bayesOpt_bounds = NULL, bayesOpt_initPoints = 4, bayesOpt_step = 1,
                       filter = NULL, transformation = NULL, gridsearch_cv = F, gridsearch_cv.min = T, hyper_grid = NULL) {

  options <- c()
  options$distribution <- distribution
  options$step_log <- step_log
  options$step_normalize <- step_normalize
  options$hidden <- hidden
  options$dropout.hidden <- dropout.hidden
  options$activation.hidden <- activation.hidden
  options$activation.output <- activation.output
  options$activation.output.cann <- activation.output.cann
  options$use_bias <- use_bias
  options$loss <- loss
  options$optimizer <- optimizer
  options$epochs <- epochs
  options$batch_size <- batch_size
  options$metrics <- metrics
  options$validation_split <- validation_split
  options$monitor <- monitor
  options$patience <- patience
  options$batch_normalization <- batch_normalization
  options$fixed.cann <- fixed.cann
  options$family_for_glm <- family_for_glm
  options$verbose <- verbose
  options$bayesOpt <- bayesOpt
  options$bayesOpt.min <- bayesOpt_min
  options$bayesOpt_iters_n <- bayesOpt_iters_n
  options$bayesOpt_bounds <- bayesOpt_bounds
  options$bayesOpt_initPoints <- bayesOpt_initPoints
  options$nfolds <- nfolds
  options$use_embedding <- use_embedding
  options$formula.glm <- formula.glm
  options$bias_regularization <- bias_regularization
  options$embedding_var <- embedding_var
  options$embedding_var.glm <- embedding_var.glm
  options$bayesOpt_step <- bayesOpt_step
  options$gridsearch_cv <- gridsearch_cv
  options$gridsearch_cv.min <- gridsearch_cv.min
  options$hyper_grid <- hyper_grid
  options$shuffle <- shuffle

  if(is.null(options$hidden)) {
    if(!is.null(dropout.hidden)) stop('If hidden is NULL, so should be dropout.hidden.')
    if(!is.null(activation.hidden)) stop('If hidden is NULL, so should be activation.hidden.')
  }
  else {
    if(!is.null(options$dropout.hidden)) {
      if(length(options$hidden) != length(options$dropout.hidden)) {
        stop('The length of hidden and dropout.hidden should match.')
      }
    }
    if(is.null(options$activation.hidden)) options$activation.hidden <- rep('relu',length(options$hidden))
    if(length(options$hidden) != length(options$activation.hidden)) {
      stop('The length of hidden and activation.hidden should match.')
    }
  }

  if(is.null(options$bias_regularization)) {

    if(is.null(options$hidden)) {
      # If no hidden layer, no need for bias regularization
      options$bias_regularization <- FALSE
    }
    else {
      if(options$distribution == 'gamma') { # Link = log -> activation = exponential
        if(options$activation.output.cann == 'exponential')
          options$bias_regularization <- TRUE
        else
          stop('For the bias regularization to work, the CANN output layer activation function should be exponential.')
      }
      else if(options$distribution == 'bernoulli') { # Default link = logit -> activation = sigmoid
        if(options$activation.output.cann == 'sigmoid')
          options$bias_regularization <- TRUE
        else
          stop('For the bias regularization to work, the CANN output layer activation function should be sigmoid')
      }
      else if(options$distribution == 'poisson') { # Default link = log -> activation = exponential
        if(options$activation.output.cann == 'exponential')
          options$bias_regularization <- TRUE
        else
          stop('For the bias regularization to work, the CANN output layer activation function should be exponential')
      }
      else if(options$distribution == 'gaussian') { # Default link = identity -> activation = linear
        if(options$activation.output.cann == 'linear')
          options$bias_regularization <- TRUE
        else
          stop('For the bias regularization to work, the CANN output layer activation function should be linear')
      }
      else
        stop('Bias regularization is not supported (yet) for this choice of distribution.')
    }

  }

  hirem_layer(obj, name, 'cann', 'layer_cann', options, filter, transformation)

}

#' Layer estimated using AutoML (H2O)
#'
#' Adds a new layer to the hierarchical reserving model. This layer will be estimated using AutoML from the \code{h2o} package.
#'
#' @param obj The hierarchical reserving model
#' @param name Character, name of the layer. This name should match the variable name in the data set
#' @param filter Function with \itemize{
#'   \item input: Data set with same structure as the data passed to \code{hirem}
#'   \item output: TRUE/FALSE vector with same length as the number of rows in the input data set.\cr
#'         FALSE indicates that this layer is zero for the current record.
#'  }
#' @param transformation Object of class \code{hirem_transformation} specifying the transformation
#' applied before modelling this layer.
#' @export
layer_aml_h2o <- function(obj, name, distribution = 'gaussian',
                      max_models = 5, filter = NULL, transformation = NULL) {

  options <- c()
  options$max_models <- max_models
  options$distribution <- distribution

  hirem_layer(obj, name, 'aml_h2o', 'layer_aml_h2o', options, filter, transformation)
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
