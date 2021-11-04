#' ---
#' title: "Combined Actuarial Neural Networks (CANN) as layer in `hirem`"
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
#' We illustrate the use of a CANN model as layer in a hierarchical model. The
#' `keras` package is used to fit the CANN model. For more information on such
#' models, see Schelldorfer, Jürg and Wuthrich, Mario V., Nesting Classical
#' Actuarial Models into Neural Networks (January 22, 2019). Available at SSRN:
#' https://ssrn.com/abstract=3320525 or http://dx.doi.org/10.2139/ssrn.3320525.
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

#' # CANN model
#' As for the benchmark model, we consider `size` as response variable and `close`, `development_year_factor` as predictors. For the Neural Network, we opt for 2 hidden layers with 10 neurons each.
model.cann <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_cann('size', distribution = 'gamma',
             family_for_glm = Gamma(link = log),
             loss = gamma_deviance_keras,
             metrics = metric_gamma_deviance_keras,
             optimizer = optimizer_nadam(learning_rate = .01),
             validation_split = 0,
             hidden = c(10,10),
             activation.output = 'exponential',
             activation.output.cann = 'exponential',
             fixed.cann = TRUE,
             monitor = 'gamma_deviance_keras',
             patience = 20,
             epochs = 100,
             batch_size = 1000,
             filter = function(data){data$payment == 1})

model.cann <- hirem::fit(model.cann,
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year_factor')

plot(model.cann$layers$size$history, metrics = 'gamma_deviance_keras', smooth = FALSE)

#' The shape parameter estimate and standard error are close to the benchmark model.
print(model.cann$layers$size$shape)
print(model.cann$layers$size$shape.se)

#' We then compute the simulated RBNS reserves and compare them with the true
#' value. Note that a bias regularization on the fitted CANN is implemented in `layer_cann` to
#' correct for the bias (when it makes sense). For more information, see Ferrario, Andrea and Noll,
#' Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April
#' 23, 2020), p.52.
simulate_rbns(model.cann)

