rm(list=ls())
require(tidyverse)
require(hirem)
require(gbm)
require(xgboost)
require(Matrix)
require(h2o)
data("reserving_data")

######################### Models #########################

### Case 1: GLM ###

model1 <- hirem(reserving_data %>% dplyr::filter(calendar_year <= 6)) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_glm('size', Gamma(link = log),
            filter = function(data){data$payment == 1})

model1 <- fit(model1,
              close = 'close ~ factor(development_year)',
              payment = 'payment ~ close + factor(development_year)',
              size = 'size ~ close + factor(development_year)')

### Case 2: GLM + GBM ###

model2 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6)) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_gbm('size', distribution = 'gaussian',
            transformation = hirem_transformation_log,
            filter = function(data){data$payment == 1})

model2 <- fit(model2,
              close = 'close ~ factor(development_year)',
              payment = 'payment ~ close + factor(development_year)',
              size = 'size ~ close + development_year')

### Case 3: XGB ###

model3 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6)) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_xgb('size', objective = 'reg:squarederror',
            transformation = hirem_transformation_log,
            filter = function(data){data$payment == 1})

model3 <- hirem::fit(model3,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

### Case 4: XGB + DL(MLP) ###

model4 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6)) %>%
  layer_xgb('close', objective = 'binary:logistic') %>%
  layer_xgb('payment', objective= 'binary:logistic') %>%
  layer_dl('size', distribution = 'gaussian', epochs = 10,
           hidden = c(10,10), hidden_dropout_ratios = c(0.5,0.5),
           activation = 'TanhWithDropout',
           filter = function(data){data$payment == 1})

model4 <- hirem::fit(model4,
                     close = 'close ~ development_year + X1 + X2',
                     payment = 'payment ~ close + development_year + X1 + X2',
                     size = 'size ~ close + development_year + X1 + X2')

######################### Simulations #########################

update <- function(data) {
  data %>%
    dplyr::mutate(development_year = development_year + 1,
                  calendar_year = calendar_year + 1)
}

### Choose model here for simulation ###

model <- model3

### Run simulation ###

model <- register_updater(model, update)

simul <- simulate(model,
                  nsim = 10,
                  filter = function(data){dplyr::filter(data,
                                                        development_year <= 6,
                                                        close == 0)},
                  data = reserving_data %>% dplyr::filter(calendar_year == 6))

rbns_estimate <- simul %>%
  dplyr::group_by(simulation) %>%
  dplyr::summarise(rbns = sum(size))

rbns_estimate

rbns_actual <- reserving_data %>%
  dplyr::filter(calendar_year > 6) %>%
  dplyr::summarise(rbns = sum(size))

rbns_actual

