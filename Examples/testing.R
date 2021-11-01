rm(list=ls())
#options(warn=-1)
library(tidyverse)
library(data.table)
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

######################### Loading data ###################

data("reserving_data")
reserving_data <- reserving_data %>%
  mutate(development_year_factor = factor(development_year))


######################### Imports ########################

source(file='./Examples/import/functions.R')

######################### Models #########################

### Case 1: GLM ###

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
              size = 'size ~ close + factor(development_year)')

simulate_rbns(model1)

### Case 2: GLM + GBM (gaussian) ###

model2 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_gbm('size', distribution = 'gaussian', cv.folds = 6,
            filter = function(data){data$payment == 1})

model2 <- hirem::fit(model2,
              close = 'close ~ factor(development_year)',
              payment = 'payment ~ close + factor(development_year)',
              size = 'size ~ close + development_year')

simulate_rbns(model2)

### Case 2b: GLM + GBM (gamma) ###

model2b <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_gbm('size', distribution = 'gamma', cv.folds = 6,
            filter = function(data){data$payment == 1})

model2b <- hirem::fit(model2b,
              close = 'close ~ factor(development_year)',
              payment = 'payment ~ close + factor(development_year)',
              size = 'size ~ close + development_year')

simulate_rbns(model2b)

### Case 3: GLM + XGB (reg:squarederror) ###

model3 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_xgb('size', objective = 'reg:squarederror',
            eval_metric = 'rmse',
            eta = 0.01,
            nrounds = 1000,
            early_stopping_rounds = 20,
            max_depth = 6,
            verbose = F,
            filter = function(data){data$payment == 1})

model3 <- hirem::fit(model3,
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year_factor')

simulate_rbns(model3)

### Case 3b: GLM + XGB (reg:squarederror + cross-validation) ###

hyper_grid <- expand.grid(
  eta = 0.01,
  max_depth = c(4,5,6),
  min_child_weight = 1000,
  subsample = c(.5,.8,1),
  colsample_bytree = c(.5,.8,1),
  gamma = c(0,.1),
  lambda = c(0,.1),
  alpha = c(0,.1)
)

model3b <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_xgb('size', objective = 'reg:squarederror',
            eval_metric = 'rmse', nfolds = 6, hyper_grid = hyper_grid,
            nrounds = 1000,
            early_stopping_rounds = 20,
            verbose = F,
            transformation = hirem_transformation_log,
            filter = function(data){data$payment == 1})

model3b <- hirem::fit(model3b,
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year')

simulate_rbns(model3b)

### Case 3c: GLM + XGB (reg:gamma + gamma-deviance) ###

model3c <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_xgb('size', objective = 'reg:gamma',
            eval_metric = 'gamma-deviance',
            eta = 0.01,
            nrounds = 1500,
            max_depth = 20,
            verbose = F,
            filter = function(data){data$payment == 1})

model3c <- hirem::fit(model3c,
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year_factor')

simulate_rbns(model3c)

### Case 4: GLM + MLP (h2o) ###

model4 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_mlp_h2o('size', distribution = 'gaussian',
                epochs = 1,
                nfolds = 6,
                hidden = c(10,30,50,30,10),
                hidden_dropout_ratios = rep(.01,5),
                activation = 'RectifierWithDropout',
                filter = function(data){data$payment == 1})

model4 <- hirem::fit(model4,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

simulate_rbns(model4)

### Case 5: GLM + MLP (h2o) ###

model5 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_mlp_h2o('size', distribution = 'gaussian',
                epochs = 10,
                hidden = c(10,10),
                hidden_dropout_ratios = c(0.1,0.1),
                activation = 'TanhWithDropout',
                filter = function(data){data$payment == 1})

model5 <- hirem::fit(model5,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

simulate_rbns(model5)

### Case 6: GLM + AutoML (h2o) ###

model6 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_aml_h2o('size', distribution = 'gaussian',
            filter = function(data){data$payment == 1})

model6 <- hirem::fit(model6,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

simulate_rbns(model6)

### Case 7: GLM + MLP (keras) with weight initialization by an homogeneous GLM ###

model7 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_mlp_keras('size', distribution = 'gamma',
                  loss = gamma_deviance_keras,
                  metrics = metric_gamma_deviance_keras,
                  optimizer = optimizer_nadam(learning_rate = 0.01),
                  validation_split = 0.2,
                  hidden = c(30,20,10),
                  activation.output = 'exponential',
                  batch_normalization = F,
                  epochs = 100,
                  scale = F,
                  batch_size = 1000,
                  monitor = 'val_gamma_deviance_keras',
                  patience = 20,
                  filter = function(data){data$payment == 1})

model7 <- hirem::fit(model7,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

simulate_rbns(model7)

### Case 7b: GLM + MLP (gamma deviance) ###

model7b <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_mlp_keras('size', distribution = 'gamma',
                  log = F,
                  normalize = T,
                  loss = gamma_deviance_keras,
                  metrics = metric_gamma_deviance_keras,
                  optimizer = optimizer_nadam(),
                  validation_split = 0.2,
                  hidden = NULL,
                  activation.output = 'exponential',
                  batch_normalization = F,
                  epochs = 100,
                  scale = F,
                  batch_size = 1000,
                  monitor = 'val_gamma_deviance_keras',
                  patience = 20,
                  filter = function(data){data$payment == 1})

model7b <- hirem::fit(model7b,
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year_factor')

simulate_rbns(model7b)

### Case 8: MLP (keras) with weight initialization by an homogeneous GLM ###

model8 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_mlp_keras('payment', distribution = 'bernoulli', loss = 'binary_crossentropy',
                  optimizer = 'nadam', validation_split = .2,
                  hidden = c(70,60,50), dropout.hidden = rep(.01,3), activation.hidden = rep('relu',3),
                  activation.output = 'softmax', epochs = 100, batch_size = 1000, family_for_init = binomial(link = logit)) %>%
  layer_mlp_keras('size', distribution = 'gaussian', loss = 'mse',
                  optimizer = 'nadam', validation_split = .2,
                  hidden = c(70,50,40), dropout.hidden = rep(.01,3), activation.hidden = rep('relu',3),
                  activation.output = 'linear', epochs = 100, batch_size = 1000, family_for_init = Gamma(link = log),
                  filter = function(data){data$payment == 1})

model8 <- hirem::fit(model8,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

simulate_rbns(model8)

### Case 9: GLM + CANN ###

model9 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_cann('size', distribution = 'gaussian', family_for_glm = Gamma(link = log), loss = 'mse',
                  optimizer = 'nadam', validation_split = .2,
                  hidden = c(60,50,40,30), dropout.hidden = rep(.01,4),
                  activation.hidden = rep('relu',4), activation.output = 'linear',
                  activation.output.cann = 'linear', fixed.cann = TRUE,
                  epochs = 100, batch_size = 10000,
                  filter = function(data){data$payment == 1})

model9 <- hirem::fit(model9,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

simulate_rbns(model9)

### Case 10: GLM + CANN (gamma with custom metric) ###

model10 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_cann('size', distribution = 'gamma', family_for_glm = Gamma(link = log),
             loss = gamma_deviance_keras, metrics = metric_gamma_deviance_keras,
             optimizer = 'nadam', validation_split = .2,
             hidden = c(60,50,40,30), dropout.hidden = rep(.01,4),
             activation.hidden = rep('relu',4), activation.output = 'linear',
             activation.output.cann = 'linear', fixed.cann = TRUE,
             epochs = 100, batch_size = 10000,
             filter = function(data){data$payment == 1})

model10 <- hirem::fit(model10,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

simulate_rbns(model10)

### Case 10: GLM + CANN (gamma with custom metric) ###

model10 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_cann('size', distribution = 'gamma', family_for_glm = Gamma(link = log),
             loss = gamma_deviance_keras, metrics = metric_gamma_deviance_keras,
             optimizer = 'nadam', validation_split = .2,
             hidden = c(60,50,40,30), dropout.hidden = rep(.01,4),
             activation.hidden = rep('relu',4), activation.output = 'linear',
             activation.output.cann = 'linear', fixed.cann = TRUE,
             epochs = 100, batch_size = 10000,
             filter = function(data){data$payment == 1})

model10 <- hirem::fit(model10,
                      close = 'close ~ development_year',
                      payment = 'payment ~ close + development_year',
                      size = 'size ~ close + development_year')

simulate_rbns(model10)
