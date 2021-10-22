test_that("3xlayer_glm", {

library(tidyverse)
library(hirem)
#library(devtools)
#devtools::install_github("harrysouthworth/gbm")
library(gbm)
library(xgboost)
library(Matrix)
library(h2o)
library(data.table)
data("reserving_data")

############### Imports ###############

simulate_rbns <- function(model, nsim = 5) {

  update <- function(data) {
    data %>%
      dplyr::mutate(development_year = development_year + 1,
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

############### 3x layer_glm ###############

model1 <- hirem(reserving_data) %>%
    split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
               validation = .7, cv_fold = 6) %>%
    layer_glm('close', binomial(link = logit)) %>%
    layer_glm('payment', binomial(link = logit)) %>%
    layer_glm('size', Gamma(link = log),
              filter = function(data){data$payment == 1})

expect_silent(model1)

model1 <- fit(model1,
              close = 'close ~ factor(development_year)',
              payment = 'payment ~ close + factor(development_year)',
              size = 'size ~ close + factor(development_year)')

expect_silent(model1)

expect_error(simulate_rbns(model1), NA)

})
