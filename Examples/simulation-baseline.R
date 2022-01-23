# Setup
options(dplyr.summarise.inform = FALSE)
options(warn = -1)

require(tidyverse)
require(xgboost)
library(ParBayesianOptimization)
library(doParallel)
require(Matrix)
require(data.table)
require(ggplot2)
require(keras)
require(tensorflow)
require(recipes)
#install.packages('devtools')
#devtools::install_github('cnoza/hirem')
require(hirem)

gamma_deviance_keras <- function(yobs, yhat) {
  K <- backend()
  2*K$mean((yobs-yhat)/yhat - K$log(yobs/yhat))
}

metric_gamma_deviance_keras <- keras::custom_metric("gamma_deviance_keras", function(yobs, yhat) {
  gamma_deviance_keras(yobs, yhat)
})

# Update function
update <- function(data) {
  data$dev.year <- data$dev.year + 1
  data$dev.year.fact <- factor(data$dev.year, levels = 1:9)

  data$calendar.year <- data$calendar.year + 1

  data$monthDev12[data$dev.year > 3] <- 'dev.year.over.3'
  data$devYearMonth <- factor(paste(data$dev.year, data$monthDev12, sep = '-'))

  data
}

seeds <- c()

obs_open_total <- c()
cl_open_total  <- c()
glm_open_total <- c()
gbm_open_total <- c()
xgb_open_total <- c()
dnn_open_total <- c()
cann_open_total <- c()

obs_pay_total <- c()
cl_pay_total  <- c()
glm_pay_total <- c()
gbm_pay_total <- c()
xgb_pay_total <- c()
dnn_pay_total <- c()
cann_pay_total <- c()

obs_size_total <- c()
cl_size_total  <- c()
glm_size_total <- c()
gbm_size_total <- c()
xgb_size_total <- c()
dnn_size_total <- c()
cann_size_total <- c()

max_sim <- 2

for(seed in 1:max_sim) {

  seeds[seed] <- seed

  # The complete portfolio
  reserving_data <- simulate_scenario_baseline(seed = seed, n = 125000)

  # We unset the seed
  set.seed(NULL)

  # Creating the interaction effect
  reserving_data$monthDev12 <- as.character(reserving_data$rep.month)
  reserving_data$monthDev12[reserving_data$dev.year > 3] <- 'dev.year.over.3'
  reserving_data$devYearMonth <- factor(paste(reserving_data$dev.year, reserving_data$monthDev12, sep = '-'))

  # Observed and prediction data set
  observed_data   <- reserving_data %>% filter(calendar.year <= 9)
  prediction_data <- reserving_data %>% filter(calendar.year > 9)

  # Calculating the weights
  reported_claims <- observed_data %>%
    dplyr::filter(dev.year == 1) %>%
    group_by(rep.year) %>%
    dplyr::summarise(count = n()) %>%
    pull(count)

  denominator <- tail(rev(cumsum(reported_claims)), -1)
  numerator <- head(cumsum(rev(reported_claims)), -1)
  weight <- c(10^(-6), numerator / denominator)

  names(weight) <- paste0('dev.year',1:9)

  # Model specifications hierarchical GLM
  model_glm  <- hirem(reserving_data) %>%
    split_data(observed = function(df) df %>% filter(calendar.year <= 9, open == 1)) %>%
    layer_glm(name = 'settlement', 'family' = binomial(link = cloglog)) %>%
    layer_glm(name = 'payment', 'family' = binomial(link = logit)) %>%
    layer_glm(name = 'size', 'family' = Gamma(link = 'log'),
              filter = function(x){x$payment == 1})

  # Formulae hierarchical GLM layers - model calibration
  formula_settle_glm <- "settlement ~ type + dev.year.fact"
  formula_pay_glm    <- "payment ~ settlement + type + devYearMonth"
  formula_size_glm   <- "size ~ devYearMonth + type + settlement + rep.month"

  # Fitting the hierarchical GLM
  model_glm <- hirem::fit(model_glm,
                          weights = weight,
                          weight.var = 'dev.year',
                          balance.var = 'dev.year',
                          settlement = formula_settle_glm,
                          payment = formula_pay_glm,
                          size = formula_size_glm)

  # Results of hierarchical model calibration
  gbm_param_settle <- list('n.trees' = 225, 'interaction.depth' = 1, 'shrinkage' = 0.05)
  gbm_param_pay    <- list('n.trees' = 125, 'interaction.depth' = 3, 'shrinkage' = 0.05)
  gbm_param_size   <- list('n.trees' = 700, 'interaction.depth' = 1, 'shrinkage' = 0.05)

  # Model specifications
  model_gbm <- hirem(reserving_data) %>%
    split_data(observed = function(df) df %>% filter(calendar.year <= 9, open == 1)) %>%
    layer_gbm('settlement', distribution = 'bernoulli', bag.fraction = 0.75, n.minobsinnode = 100,
              n.trees = gbm_param_settle$n.trees, interaction.depth = gbm_param_settle$interaction.depth,
              shrinkage = gbm_param_settle$shrinkage, select_trees = 'last') %>%
    layer_gbm('payment', distribution = 'bernoulli', bag.fraction = 0.75, n.minobsinnode = 100,
              n.trees = gbm_param_pay$n.trees, interaction.depth = gbm_param_pay$interaction.depth,
              shrinkage = gbm_param_pay$shrinkage, select_trees = 'last') %>%
    layer_gbm('size', distribution = 'gamma', bag.fraction = 0.75, n.minobsinnode = 100,
              n.trees = gbm_param_size$n.trees, interaction.depth = gbm_param_size$interaction.depth,
              shrinkage = gbm_param_size$shrinkage, select_trees = 'last',
              filter = function(data){data$payment == 1})

  # Covariates
  covariates_gbm <- c('type', 'dev.year.fact', 'rep.month', 'rep.year.fact', 'rep.delay', 'calendar.year')

  # Fitting the hierarchical GBM
  model_gbm <- hirem::fit(model_gbm,
                          weights = weight,
                          weight.var = 'dev.year',
                          balance.var = 'dev.year',
                          settlement = paste0('settlement ~ 1 + ', paste0(covariates_gbm, collapse = ' + ')),
                          payment = paste0('payment ~ 1 + ', paste0(c(covariates_gbm, 'settlement'), collapse = ' + ')),
                          size = paste0('size ~ 1 + ', paste0(c(covariates_gbm,'settlement'), collapse = ' + ')))

  # Results of hierarchical model calibration
  xgb_param_settle <- list('eta' =   1, 'max_depth' = 1, 'nrounds' =  10, 'subsample' = 1)
  xgb_param_pay    <- list('eta' =  .5, 'max_depth' = 3, 'nrounds' =  10, 'subsample' = 1)
  xgb_param_size   <- list('eta' = .05, 'max_depth' = 1, 'nrounds' = 690, 'subsample' = .75)

  # Model specifications
  model_xgb <- hirem(reserving_data) %>%
    split_data(observed = function(df) df %>% filter(calendar.year <= 9, open == 1)) %>%
    layer_xgb('settlement', objective = 'binary:logistic', eval_metric = 'logloss', grow_policy = 'lossguide',
              eta = xgb_param_settle$eta, max_depth = xgb_param_settle$max_depth,
              nrounds = xgb_param_settle$nrounds, subsample = xgb_param_settle$subsample) %>%
    layer_xgb('payment', objective = 'binary:logistic', eval_metric = 'logloss', grow_policy = 'lossguide',
              eta = xgb_param_pay$eta, max_depth = xgb_param_pay$max_depth,
              nrounds = xgb_param_pay$nrounds, subsample = xgb_param_pay$subsample) %>%
    layer_xgb('size', objective = 'reg:gamma',
              eval_metric = 'gamma-deviance', grow_policy = 'lossguide',
              eta = xgb_param_size$eta, max_depth = xgb_param_size$max_depth,
              nrounds = xgb_param_size$nrounds, subsample = xgb_param_size$subsample,
              filter = function(data){data$payment == 1})

  # Covariates
  covariates_xgb <- c('type', 'dev.year.fact', 'rep.month', 'rep.year.fact', 'rep.delay', 'calendar.year')

  # Fitting the hierarchical XGB
  model_xgb <- hirem::fit(model_xgb,
                          weights = weight,
                          weight.var = 'dev.year',
                          balance.var = 'dev.year',
                          settlement = paste0('settlement ~ ', paste0(covariates_xgb, collapse = ' + ')) ,
                          payment = paste0('payment ~ ', paste0(c(covariates_xgb, 'settlement'), collapse = ' + ')),
                          size = paste0('size ~ ', paste0(c(covariates_xgb,'settlement'), collapse = ' + ')))

  # Model specifications
  model_dnn <-  hirem(reserving_data) %>%
    split_data(observed = function(df) df %>% filter(calendar.year <= 9, open == 1)) %>%
    layer_dnn('settlement', distribution = 'bernoulli',
              bias_regularization = T,
              hidden = c(40,30,40),
              loss = 'binary_crossentropy',
              metrics = 'binary_crossentropy',
              optimizer = optimizer_nadam(),
              validation_split = .3,
              activation.output = 'sigmoid',
              family_for_init = binomial(), # default link = logit
              epochs = 100,
              batch_size = 1000,
              monitor = 'val_binary_crossentropy',
              patience = 20) %>%
    layer_dnn('payment', distribution = 'bernoulli',
              bias_regularization = T,
              hidden = c(30,40,20),
              loss = 'binary_crossentropy',
              metrics = 'binary_crossentropy',
              optimizer = optimizer_nadam(),
              validation_split = .3,
              activation.output = 'sigmoid',
              family_for_init = binomial(), # default link = logit
              epochs = 100,
              batch_size = 1000,
              monitor = 'val_binary_crossentropy',
              patience = 20) %>%
    layer_dnn('size', distribution = 'gamma',
              bias_regularization = T,
              hidden = c(50,40,60),
              loss = gamma_deviance_keras,
              metrics = metric_gamma_deviance_keras,
              optimizer = optimizer_nadam(),
              validation_split = .3,
              activation.output = 'exponential',
              family_for_init = Gamma(link=log), # default link = inverse
              epochs = 100,
              batch_size = 1000,
              monitor = 'val_gamma_deviance_keras',
              patience = 20,
              filter = function(data){data$payment == 1})

  # Covariates
  covariates_dnn <- c('type', 'dev.year.fact', 'rep.month', 'rep.year.fact', 'rep.delay', 'calendar.year')

  # Fitting the hierarchical DNN
  model_dnn <- hirem::fit(model_dnn,
                          balance.var = 'dev.year',
                          settlement = paste0('settlement ~ ', paste0(covariates_dnn, collapse = ' + ')),
                          payment = paste0('payment ~ ', paste0(c(covariates_dnn, 'settlement'), collapse = ' + ')),
                          size = paste0('size ~ ', paste0(c(covariates_dnn,'settlement'), collapse = ' + ')))

  # Model specifications
  model_cann <-  hirem(reserving_data) %>%
    split_data(observed = function(df) df %>% filter(calendar.year <= 9, open == 1)) %>%
    layer_cann('settlement', distribution = 'bernoulli',
               hidden = c(50,50,50),
               formula.glm = formula_settle_glm,
               bias_regularization = T,
               family_for_glm = binomial(),
               loss = 'binary_crossentropy',
               metrics = 'binary_crossentropy',
               optimizer = optimizer_nadam(),
               validation_split = .3,
               activation.output = 'linear',
               activation.output.cann = 'sigmoid',
               fixed.cann = T,
               monitor = 'val_binary_crossentropy',
               patience = 20,
               epochs = 100,
               batch_size = 1000) %>%
    layer_cann('payment', distribution = 'bernoulli',
               hidden = c(30,30,50),
               formula.glm = formula_pay_glm,
               bias_regularization = T,
               family_for_glm = binomial(),
               loss = 'binary_crossentropy',
               metrics = 'binary_crossentropy',
               optimizer = optimizer_nadam(),
               validation_split = .3,
               activation.output = 'linear',
               activation.output.cann = 'sigmoid',
               fixed.cann = T,
               monitor = 'val_binary_crossentropy',
               patience = 20,
               epochs = 100,
               batch_size = 1000) %>%
    layer_cann('size', distribution = 'gamma',
               hidden = c(40,20,20),
               formula.glm = formula_size_glm,
               bias_regularization = T,
               family_for_glm = Gamma(link=log),
               loss = gamma_deviance_keras,
               metrics = metric_gamma_deviance_keras,
               optimizer = optimizer_nadam(),
               validation_split = .3,
               activation.output = 'linear',
               activation.output.cann = 'exponential',
               fixed.cann = T,
               monitor = 'val_gamma_deviance_keras',
               patience = 20,
               epochs = 100,
               batch_size = 1000,
               filter = function(data){data$payment == 1})

  # Covariates
  covariates_cann <- c('type', 'dev.year.fact', 'rep.month', 'rep.year.fact', 'rep.delay', 'calendar.year')

  # Fitting the hierarchical CANN
  model_cann <- hirem::fit(model_cann,
                           balance.var = 'dev.year',
                           settlement = paste0('settlement ~ ', paste0(covariates_cann, collapse = ' + ')),
                           payment = paste0('payment ~ ', paste0(c(covariates_cann, 'settlement'), collapse = ' + ')),
                           size = paste0('size ~ ', paste0(c(covariates_cann,'settlement'), collapse = ' + ')))

  model_glm <- register_updater(model_glm, update)
  model_gbm <- register_updater(model_gbm, update)
  model_xgb <- register_updater(model_xgb, update)
  model_dnn <- register_updater(model_dnn, update)
  model_cann <- register_updater(model_cann, update)

  nsim <- 50

  simul_glm <- simulate(model_glm,
                        nsim = nsim,
                        filter = function(data){dplyr::filter(data, dev.year <= 9, settlement == 0)},
                        data = model_glm$data_observed %>% dplyr::filter(calendar.year == 9),
                        balance.correction = TRUE)

  simul_gbm <- simulate(model_gbm,
                        nsim = nsim,
                        filter = function(data){dplyr::filter(data, dev.year <= 9, settlement == 0)},
                        data = model_gbm$data_observed %>% dplyr::filter(calendar.year == 9),
                        balance.correction = TRUE)

  simul_xgb <- simulate(model_xgb,
                        nsim = nsim,
                        filter = function(data){dplyr::filter(data, dev.year <= 9, settlement == 0)},
                        data = model_xgb$data_observed %>% dplyr::filter(calendar.year == 9),
                        balance.correction = TRUE)

  simul_dnn <- simulate(model_dnn,
                        nsim = nsim,
                        filter = function(data){dplyr::filter(data, dev.year <= 9, settlement == 0)},
                        data = model_dnn$data_observed %>% dplyr::filter(calendar.year == 9),
                        balance.correction = TRUE)

  simul_cann <- simulate(model_cann,
                         nsim = nsim,
                         filter = function(data){dplyr::filter(data, dev.year <= 9, settlement == 0)},
                         data = model_cann$data_observed %>% dplyr::filter(calendar.year == 9),
                         balance.correction = TRUE)

  # Incremental run-off triangles
  triangle_open    <- construct_triangle(data = observed_data %>% filter(open == 1), group.var1 = 'rep.year',
                                         group.var2 = 'dev.year', value = 'open', cumulative = FALSE)
  triangle_payment <- construct_triangle(data = observed_data, group.var1 = 'rep.year',
                                         group.var2 = 'dev.year', value = 'payment', cumulative = FALSE)
  triangle_size    <- construct_triangle(data = observed_data, group.var1 = 'rep.year',
                                         group.var2 = 'dev.year', value = 'size', cumulative = FALSE)

  # Number of open claims in the year following the evaluation date
  settle.evalyear <- observed_data %>%
    filter(open == 1, calendar.year == 9) %>%
    group_by(rep.year, dev.year) %>%
    summarise(settlement = sum(settlement))

  # The number of open claims in the year after the evaluation date
  triangle_open[row(triangle_open) + col(triangle_open) == 11] <-
    (triangle_open[row(triangle_open) + col(triangle_open) == 10] - rev(settle.evalyear$settlement))[1:8]

  # Chain ladder predictions
  cl_open <- chainLadder_open(triangle_open)
  cl_pay  <- chainLadder(triangle_payment, is_cumulatif = FALSE)
  cl_size <- chainLadder(triangle_size, is_cumulatif = FALSE)

  # Predictions
  obs_open_total[seed] <- prediction_data %>% filter(calendar.year != 10) %>% summarise(Total = sum(open)) %>% pull(Total)
  cl_open_total[seed]  <- sum(cl_open)
  glm_open_total[seed] <- simul_glm %>% filter(calendar.year != 10) %>% summarise(Total = sum(open)/nsim) %>% pull(Total)
  gbm_open_total[seed] <- simul_gbm %>% filter(calendar.year != 10) %>% summarise(Total = sum(open)/nsim) %>% pull(Total)
  xgb_open_total[seed] <- simul_xgb %>% filter(calendar.year != 10) %>% summarise(Total = sum(open)/nsim) %>% pull(Total)
  dnn_open_total[seed] <- simul_dnn %>% filter(calendar.year != 10) %>% summarise(Total = sum(open)/nsim) %>% pull(Total)
  cann_open_total[seed] <- simul_cann %>% filter(calendar.year != 10) %>% summarise(Total = sum(open)/nsim) %>% pull(Total)

  # Predictions
  obs_pay_total[seed] <- prediction_data %>% summarise(Total = sum(payment)) %>% pull(Total)
  cl_pay_total[seed]  <- sum(cl_pay)
  glm_pay_total[seed] <- simul_glm %>% summarise(Total = sum(payment)/nsim) %>% pull(Total)
  gbm_pay_total[seed] <- simul_gbm %>% summarise(Total = sum(payment)/nsim) %>% pull(Total)
  xgb_pay_total[seed] <- simul_xgb %>% summarise(Total = sum(payment)/nsim) %>% pull(Total)
  dnn_pay_total[seed] <- simul_dnn %>% summarise(Total = sum(payment)/nsim) %>% pull(Total)
  cann_pay_total[seed] <- simul_cann %>% summarise(Total = sum(payment)/nsim) %>% pull(Total)

  # Predictions
  obs_size_total[seed] <- prediction_data %>% summarise(Total = sum(size)) %>% pull(Total)
  cl_size_total[seed]  <- sum(cl_size)
  glm_size_total[seed] <- simul_glm %>% summarise(Total = sum(size)/nsim) %>% pull(Total)
  gbm_size_total[seed] <- simul_gbm %>% summarise(Total = sum(size)/nsim) %>% pull(Total)
  xgb_size_total[seed] <- simul_xgb %>% summarise(Total = sum(size)/nsim) %>% pull(Total)
  dnn_size_total[seed] <- simul_dnn %>% summarise(Total = sum(size)/nsim) %>% pull(Total)
  cann_size_total[seed] <- simul_cann %>% summarise(Total = sum(size)/nsim) %>% pull(Total)

  rm(simul_glm)
  rm(simul_gbm)
  rm(simul_xgb)
  rm(simul_dnn)
  rm(simul_cann)

}

results <- data.frame(
  seeds,
  obs_open_total, obs_pay_total, obs_size_total,
  cl_open, cl_pay, cl_size,
  glm_open_total, glm_pay_total, glm_size_total,
  gbm_open_total, gbm_pay_total, gbm_size_total,
  xgb_open_total, xgb_pay_total, xgb_size_total,
  dnn_open_total, dnn_pay_total, dnn_size_total,
  cann_open_total, cann_pay_total, cann_size_total
)

write_csv(results,'./results-simulation-baseline.csv')
