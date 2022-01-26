

#' Layer estimated using a deep feed-forward neural network model with Keras
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
                            filter = NULL, transformation = NULL, gridsearch_cv = F, gridsearch_cv.min = T, hyper_grid = NULL, random_trials = 0, one_hot = TRUE) {

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
  options$random_trials <- random_trials

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
