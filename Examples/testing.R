rm(list=ls())
#options(warn=-1)
library(tidyverse)
library(hirem)
library(devtools)
#devtools::install_github("harrysouthworth/gbm")
library(gbm)
library(xgboost)
library(Matrix)
library(h2o)
library(data.table)
data("reserving_data")

######################### Imports #########################

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

model1 <- fit(model1,
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
            transformation = hirem_transformation_log,
            filter = function(data){data$payment == 1})

model2 <- fit(model2,
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
            transformation = hirem_transformation_log,
            filter = function(data){data$payment == 1})

model2b <- fit(model2b,
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
            transformation = hirem_transformation_log,
            filter = function(data){data$payment == 1})

model3 <- hirem::fit(model3,
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year')

simulate_rbns(model3)

### Case 3b: GLM + XGB (reg:squarederror + nfolds) ###

model3b <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_xgb('size', objective = 'reg:squarederror',
            eval_metric = 'rmse', nfolds = 6,
            nrounds = 500,
            early_stopping_rounds = 10,
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
            transformation = hirem_transformation_log,
            filter = function(data){data$payment == 1})

model3c <- hirem::fit(model3c,
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year')

simulate_rbns(model3c)

### Case 4: GLM + DL(MLP) ###

model4 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dl('size', distribution = 'gaussian', epochs = 20, nfolds = 6,
           hidden = c(100,100,100), hidden_dropout_ratios = c(0.01,0.01,0.01),
           activation = 'RectifierWithDropout',
           filter = function(data){data$payment == 1})

model4 <- hirem::fit(model4,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

simulate_rbns(model4)

### Case 5: GLM + DL(MLP) ###

model5 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dl('size', distribution = 'gaussian', epochs = 10,
           hidden = c(10,10), hidden_dropout_ratios = c(0.1,0.1),
           activation = 'TanhWithDropout',
           filter = function(data){data$payment == 1})

model5 <- hirem::fit(model5,
                     close = 'close ~ development_year + X1 + X2',
                     payment = 'payment ~ close + development_year + X1 + X2',
                     size = 'size ~ close + development_year + X1 + X2')

simulate_rbns(model5)

### Case 6: GLM + AutoML (h2o) ###

model6 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_aml('size', distribution = 'gaussian',
            filter = function(data){data$payment == 1})

model6 <- hirem::fit(model6,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

simulate_rbns(model6)

