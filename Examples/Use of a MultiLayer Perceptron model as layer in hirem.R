#' ---
#' title: "MultiLayer Perceptron models as layer in `hirem`"
#' author: "Christophe Nozaradan"
#' output:
#'    html_document:
#'      toc: false
#'      toc_depth: 2
#'      toc_float:
#'        collapsed: true
#'        smooth_scroll: true
#'      number_sections: false
#'      fig_caption: true
#'      df_print: paged
#'    fontsize: 11pt
#' bibliography: references.bib
#' ---

#+ setup, include=FALSE
rm(list=ls())
defaultW <- getOption("warn")
options(warn = -1)

#' # Setup
#' We illustrate the use of a MultiLayer Perceptron (MLP) model as layer in a hierarchical model. The `keras` package is used to fit the MLP model.
#' We first load the required packages.
#+ message=FALSE
library(tidyverse)
library(data.table)
library(ggplot2)
library(hirem)
library(devtools)
#devtools::install_github("harrysouthworth/gbm")
library(keras)
library(tensorflow)
library(recipes)

#' We load and prepare the data. We use the data provided in the hirem package.
#+ message=FALSE
data("reserving_data")
reserving_data <- reserving_data %>%
  mutate(development_year_factor = factor(development_year))

#' We set the seed for reproducibility.
#+ message=FALSE
set.seed(265)
set_random_seed(265)

#' We here define a function that computes the simulated RBNS reserve based on the choice of a given model.
#+ message=FALSE
simulate_rbns <- function(model, nsim = 5) {

  levels <- unique(reserving_data$development_year)

  update <- function(data) {
    data %>%
      dplyr::mutate(development_year = development_year + 1,
                    development_year_factor = factor(development_year, levels=levels),
                    calendar_year = calendar_year + 1)
  }

  model <- register_updater(model, update)

  simul <- simulate(model,
                    nsim = nsim,
                    filter = function(data){dplyr::filter(data,
                                                          development_year <= 6,
                                                          close == 0)},
                    data = reserving_data %>% dplyr::filter(calendar_year == 6))

  rbns_estimate <- simul %>%
    dplyr::group_by(simulation) %>%
    dplyr::summarise(rbns_simulated = sum(size))

  print(rbns_estimate)

  rbns_actual <- reserving_data %>%
    dplyr::filter(calendar_year > 6) %>%
    dplyr::summarise(rbns_actual = sum(size))

  print(rbns_actual)

}

#' We will use the gamma deviance as custom loss function and metric in `keras`:
gamma_deviance_keras <- function(yobs, yhat) {
  K <- backend()
  2*K$sum((yobs-yhat)/yhat - K$log(yobs/yhat))
}

metric_gamma_deviance_keras <- keras::custom_metric("gamma_deviance_keras", function(yobs, yhat) {
  gamma_deviance_keras(yobs, yhat)
})

#' # Shallow MLP layer
#' As first example, we show that an homogeneous GLM (gamma, log link) is
#' equivalent to an MLP with no hidden layer when the gamma deviance is used as loss
#' function and the exponential as activation function.
model0 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_mlp_keras('size', distribution = 'gamma',
                  step_log = F,
                  step_normalize = F,
                  loss = gamma_deviance_keras,
                  use_bias = FALSE,
                  metrics = metric_gamma_deviance_keras,
                  optimizer = optimizer_nadam(learning_rate = .01),
                  validation_split = 0,
                  hidden = NULL,
                  activation.output = 'exponential',
                  batch_normalization = F,
                  epochs = 100,
                  batch_size = 1000,
                  monitor = 'gamma_deviance_keras',
                  patience = 20,
                  filter = function(data){data$payment == 1})

model0 <- hirem::fit(model0,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ 1')

plot(model0$layers$size$history, metrics = 'gamma_deviance_keras', smooth = FALSE)

#' Let's compare with the homogeneous GLM (gamma log link):
glm.hom <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('size', Gamma(link = log),
            filter = function(data){data$payment == 1})

glm.hom <- hirem::fit(glm.hom,
                      size = 'size ~ 1')

#' The obtained coefficients are (almost) identical:
print(glm.hom$layers$size$fit$coefficients)
print(model0$layers$size$fit$weights)

#' The shape parameter of the gamma distribution are (almost) identical:
print(glm.hom$layers$size$shape)
print(model0$layers$size$shape)

#' # Benchmark model
#' For what follows, we use the following model as benchmark.
model1 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_glm('size', Gamma(link = log),
            filter = function(data){data$payment == 1})

model1 <- hirem::fit(model1,
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year_factor')

#' We obtain the shape parameter estimate and standard error.
print(model1$layers$size$shape)
print(model1$layers$size$shape.se)

#' We then compute the simulated RBNS reserves and compare them with the true value.
simulate_rbns(model1)

#' # Shallow MLP with weight initialization
#' As for the benchmark model, we consider `size` as response variable and `close`, `development_year_factor` as predictors. In the MLP, we also initialize the bias weight of the output layer with the coefficient estimate
#' of the homogeneous GLM (see parameter `family_for_init` of `layer_mlp_keras`). This allows for a faster convergence of the gradient descent algorithm.
#' For more information, see Ferrario, Andrea and Noll, Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April 23, 2020).
#' Available at SSRN: https://ssrn.com/abstract=3226852 or http://dx.doi.org/10.2139/ssrn.3226852 p.29.

model2 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_mlp_keras('size', distribution = 'gamma',
                  step_log = F,
                  step_normalize = F,
                  loss = gamma_deviance_keras,
                  metrics = metric_gamma_deviance_keras,
                  optimizer = optimizer_nadam(learning_rate = .01),
                  validation_split = 0,
                  hidden = NULL,
                  activation.output = 'exponential',
                  batch_normalization = F,
                  family_for_init = Gamma(link = log),
                  epochs = 100,
                  batch_size = 1000,
                  monitor = 'gamma_deviance_keras',
                  patience = 20,
                  filter = function(data){data$payment == 1})

model2 <- hirem::fit(model2,
                      close = 'close ~ factor(development_year)',
                      payment = 'payment ~ close + factor(development_year)',
                      size = 'size ~ close + development_year_factor')

plot(model2$layers$size$history, metrics = 'gamma_deviance_keras', smooth = FALSE)

#' The shape parameter is (almost) identical to the benchmark model:
print(model2$layers$size$shape)
print(model2$layers$size$shape.se)

print(model1$layers$size$shape)
print(model1$layers$size$shape.se)

#' We then compute the simulated RBNS reserves and compare them with the true
#' value. Note that a bias regularization on the fitted neural network is implemented in `layer_mlp_keras` to
#' correct for the bias (when it makes sense). For more information, see Ferrario, Andrea and Noll,
#' Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April
#' 23, 2020), p.52.
simulate_rbns(model2)

