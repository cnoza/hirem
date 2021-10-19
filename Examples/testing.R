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

### Case 3: GLM + XGB ###

model3 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6)) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_xgb('size', objective = 'reg:squarederror',
            eval_metric = 'rmse',
            eta = 0.01,
            nrounds = 500,
            max_depth = 6,
            verbose = F,
            transformation = hirem_transformation_log,
            filter = function(data){data$payment == 1})

model3 <- hirem::fit(model3,
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year')

### Case 4: GLM + DL(MLP) ###

model4 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6)) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dl('size', distribution = 'gaussian', epochs = 10,
           hidden = c(100,100,100), hidden_dropout_ratios = c(0.8,0.8,0.8),
           activation = 'RectifierWithDropout',
           filter = function(data){data$payment == 1})

model4 <- hirem::fit(model4,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

### Case 5: GLM + DL(MLP) ###

model5 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6)) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dl('size', distribution = 'gaussian', epochs = 10,
           hidden = c(10,10), hidden_dropout_ratios = c(0.5,0.5),
           activation = 'TanhWithDropout',
           filter = function(data){data$payment == 1})

model5 <- hirem::fit(model5,
                     close = 'close ~ development_year + X1 + X2',
                     payment = 'payment ~ close + development_year + X1 + X2',
                     size = 'size ~ close + development_year + X1 + X2')

### Case 6: GLM + AutoML (h2o) ###

model6 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6)) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_aml('size', distribution = 'gaussian',
            filter = function(data){data$payment == 1})

model6 <- hirem::fit(model6,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

######################### Simulations #########################

update <- function(data) {
  data %>%
    dplyr::mutate(development_year = development_year + 1,
                  calendar_year = calendar_year + 1)
}

### Choose model here for simulation ###

model <- model4

### Run simulation ###

model <- register_updater(model, update)

simul <- simulate(model,
                  nsim = 5,
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

