### Initialization
init <- function() {
  rm(list = ls(envir = .GlobalEnv), envir = .GlobalEnv)
  #options(warn=-1)
  library(tidyverse)
  library(data.table)
  #install.packages('devtools')
  #devtools::install_github('cnoza/hirem')
  library(hirem)
  library(devtools)
  #devtools::install_github("harrysouthworth/gbm")
  library(gbm)
  library(xgboost)
  library(Matrix)
  library(h2o)
  library(keras)
  library(tensorflow)
  library(recipes)
  set.seed(265)
  set_random_seed(265)
  source(file='./tests/import/functions.R')
  ### Loading data ###
  data("reserving_data")
  reserving_data <<- reserving_data %>%
    mutate(development_year_factor = factor(development_year))
}

### Functions used for testing purposes
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

gamma_deviance_keras <- function(yobs, yhat) {
  K <- backend()
  2*K$sum((yobs-yhat)/yhat - K$log(yobs/yhat))
}

metric_gamma_deviance_keras <- keras::custom_metric("gamma_deviance_keras", function(yobs, yhat) {
  gamma_deviance_keras(yobs, yhat)
})
