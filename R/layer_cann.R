

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
                       filter = NULL, transformation = NULL, gridsearch_cv = F, gridsearch_cv.min = T, hyper_grid = NULL, random_trials = 0) {

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
  options$random_trials <- random_trials

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
