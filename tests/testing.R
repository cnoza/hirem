### Test file ###
###
### This test file has been created to be able to easily test different cases
### implemented in the package
###
### Instructions:
###   1. Run line 12 to import the needed functions.
###   2. Go to the case you want to test and run the associated code
###      The init() function reinitialize the dataset

### Imports ###
source(file='./tests/import/functions.R')

#=========================================================================#
#                              Case 1: GLM                                #
#=========================================================================#

init()
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

print(model1$layers$size$shape)
print(model1$layers$size$shape.se)

simulate_rbns(model1)

#=========================================================================#
#                     Case 2: GLM + GBM (gaussian)                        #
#=========================================================================#

init()
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
              size = 'size ~ close + development_year_factor')

simulate_rbns(model2)

#=========================================================================#
#                     Case 2b: GLM + GBM (gamma)                          #
#=========================================================================#

init()
model2b <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_gbm('size', distribution = 'gamma', cv.folds = 5,
            filter = function(data){data$payment == 1})

model2b <- hirem::fit(model2b,
              weights = weight,
              weight.var = 'development_year_factor',
              balance.var = 'development_year_factor',
              close = 'close ~ factor(development_year)',
              payment = 'payment ~ close + factor(development_year)',
              size = 'size ~ close + development_year_factor')

print(model2b$layers$size$shape)
print(model2b$layers$size$shape.se)

simulate_rbns(model2b)

#=========================================================================#
#                   Case 3: GLM + XGB (reg:squarederror)                  #
#=========================================================================#

init()
model3 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_xgb('size', objective = 'reg:squarederror',
            eval_metric = 'rmse',
            eta = 0.05,
            nrounds = 100,
            early_stopping_rounds = 20,
            max_depth = 6,
            verbose = F,
            filter = function(data){data$payment == 1})

model3 <- hirem::fit(model3,
                     weights = weight,
                     weight.var = 'development_year_factor',
                     balance.var = 'development_year_factor',
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year_factor')

simulate_rbns(model3)

#=========================================================================#
#         Case 3b: GLM + XGB (reg:squarederror + cross-validation)        #
#=========================================================================#

init()
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

#=========================================================================#
#         Case 3c: GLM + XGB (reg:gamma + gamma-deviance)                 #
#=========================================================================#

init()

model3c <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_xgb('size', objective = 'reg:gamma',
            eval_metric = 'gamma-deviance',
            eta = 1,
            nrounds = 12,
            nthread=1,
            alpha=.960234,
            lambda=.06513692,
            gamma=0,
            max_depth = 6,
            min_child_weight = 0,
            subsample = .6944,
            colsample_bynode = 1,
            verbose = F,
            filter = function(data){data$payment == 1})

model3c <- hirem::fit(model3c,
                      weights = weight,
                      weight.var = 'development_year_factor',
                      balance.var = 'development_year_factor',
                      close = 'close ~ factor(development_year)',
                      payment = 'payment ~ close + factor(development_year)',
                      size = 'size ~ close + development_year_factor')

print(model3c$layers$size$shape)
print(model3c$layers$size$shape.se)

simulate_rbns(model3c)

#=========================================================================#
#         Case 3d: GLM + XGB (reg:gamma + gamma-deviance + bayesian opt)  #
#=========================================================================#

init()

bounds <- list(
  min_child_weight = c(0L,1000L),
  subsample = c(0.75,1),
  max_depth = c(1L,6L)
)

model3d <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_xgb('size', objective = 'reg:gamma',
            bayesOpt = T,
            bayesOpt_min = T,
            bayesOpt_iters_n = 1,
            bayesOpt_bounds = bounds,
            tree_method = 'hist',
            eta = 1,
            nfolds = 5,
            eval_metric = 'gamma-deviance',
            verbose = F,
            filter = function(data){data$payment == 1})

model3d <- hirem::fit(model3d,
                      close = 'close ~ factor(development_year)',
                      payment = 'payment ~ close + factor(development_year)',
                      size = 'size ~ close + development_year_factor')

print(model3d$layers$size$shape)
print(model3d$layers$size$shape.se)

simulate_rbns(model3d)

#=========================================================================#
#   Case 4: GLM + MLP shallow case (homogeneous, gamma, no hidden layer)  #
#=========================================================================#

# Goal: show that homogeneous GLM (gamma, log link) is equivalent to
#       shallow neural network (loss:gamma deviance, activation:exponential)

init()
model4 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dnn('size', distribution = 'gamma',
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

model4 <- hirem::fit(model4,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ 1')

# Let's compare with the homogeneous GLM (gamma log link)
glm.hom <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('size', Gamma(link = log),
            filter = function(data){data$payment == 1})

glm.hom <- hirem::fit(glm.hom,
                     size = 'size ~ 1')

# The obtained coefficients are (almost) identical:
print(glm.hom$layers$size$fit$coefficients)
print(model4$layers$size$fit$weights)

# The shape parameter of the gamma distribution are (almost) identical:
print(glm.hom$layers$size$shape)
print(model4$layers$size$shape)

#=========================================================================#
#         Case 4b: GLM + MLP shallow case (gamma, no hidden layer)        #
#=========================================================================#

# Goal:
# -----
# Show that model 1 (glm, gamma log link) is equivalent to
# shallow neural network (loss:gamma deviance, activation:exponential)
#
# For the neural network:
# -----------------------
# Initialization of the bias weight of the output layer with the coefficient estimate
# of the homogeneous GLM (parameter 'family_for_init'):
# See Ferrario, Andrea and Noll, Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April 23, 2020).
# Available at SSRN: https://ssrn.com/abstract=3226852 or http://dx.doi.org/10.2139/ssrn.3226852 p.29.

init()
model4b <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dnn('size', distribution = 'gamma',
                  step_log = F,
                  step_normalize = F,
                  loss = gamma_deviance_keras,
                  metrics = metric_gamma_deviance_keras,
                  optimizer = optimizer_nadam(learning_rate = .01),
                  validation_split = 0,
                  hidden = NULL,
                  activation.output = 'exponential',
                  batch_normalization = F,
                  family_for_init = Gamma(link=log),
                  epochs = 100,
                  batch_size = 1000,
                  monitor = 'gamma_deviance_keras',
                  patience = 20,
                  filter = function(data){data$payment == 1})

model4b <- hirem::fit(model4b,
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year_factor')

# The shape parameter is (almost) identical to model 1 (glm, gamma log link for size):
print(model4b$layers$size$shape)
print(model4b$layers$size$shape.se)

print(model1$layers$size$shape)
print(model1$layers$size$shape.se)

simulate_rbns(model4b)



#=========================================================================#
#         Case 4c: GLM + MLP (gamma, 3 hidden layers)        #
#=========================================================================#

init()
model4c <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dnn('size', distribution = 'gamma',
                  step_log = F,
                  step_normalize = F,
                  loss = gamma_deviance_keras,
                  metrics = metric_gamma_deviance_keras,
                  optimizer = optimizer_nadam(learning_rate = .01),
                  validation_split = 0,
                  hidden = c(20,15,10),
                  activation.output = 'exponential',
                  batch_normalization = F,
                  family_for_init = Gamma(link=log),
                  epochs = 100,
                  batch_size = 1000,
                  monitor = 'gamma_deviance_keras',
                  patience = 20,
                  filter = function(data){data$payment == 1})

model4c <- hirem::fit(model4c,
                      balance.var = 'development_year',
                      close = 'close ~ factor(development_year)',
                      payment = 'payment ~ close + factor(development_year)',
                      size = 'size ~ close + development_year_factor')

# The shape parameter is (almost) identical to model 1 (glm, gamma log link for size):
print(model4c$layers$size$shape)
print(model4c$layers$size$shape.se)

print(model1$layers$size$shape)
print(model1$layers$size$shape.se)

# Due to the bias regularization, we obtain RBNS simulations close to the true value:
simulate_rbns(model4c)

#=========================================================================#
#         Case 4d: GLM + MLP shallow case (bernoulli, no hidden layer)    #
#=========================================================================#

# Goal:
# -----
# Show that model 1 (glm, bernoulli logit link) is equivalent to
# shallow neural network (loss: binary_crossentropy, activation: sigmoid)
#
# For the neural network:
# -----------------------
# Initialization of the bias weight of the output layer with the coefficient estimate
# of the homogeneous GLM (parameter 'family_for_init'):
# See Ferrario, Andrea and Noll, Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April 23, 2020).
# Available at SSRN: https://ssrn.com/abstract=3226852 or http://dx.doi.org/10.2139/ssrn.3226852 p.29.

init()
model4d <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_dnn('payment', distribution = 'bernoulli',
                  step_log = F,
                  step_normalize = F,
                  loss = 'binary_crossentropy',
                  metrics = 'binary_crossentropy',
                  optimizer = optimizer_nadam(learning_rate = .01),
                  validation_split = 0,
                  hidden = NULL,
                  activation.output = 'sigmoid',
                  batch_normalization = F,
                  family_for_init = binomial,
                  epochs = 100,
                  batch_size = 1000,
                  monitor = 'binary_crossentropy',
                  patience = 20) %>%
  layer_glm('size', Gamma(link = log),
            filter = function(data){data$payment == 1})

model4d <- hirem::fit(model4d,
                      close = 'close ~ factor(development_year)',
                      payment = 'payment ~ close + development_year_factor',
                      size = 'size ~ close + development_year_factor')

# The shape parameter is (almost) identical to model 1 (glm, gamma log link for size):
print(model4d$layers$size$shape)
print(model4d$layers$size$shape.se)

print(model1$layers$size$shape)
print(model1$layers$size$shape.se)

simulate_rbns(model4d)

#=========================================================================#
#         Case 4e: GLM + MLP shallow case (gaussian, no hidden layer)    #
#=========================================================================#

# Goal:
# -----
# Show that a GLM gaussian is equivalent to
# shallow neural network (loss: mean_squared_error, activation: linear)

init()
model4e <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dnn('size', distribution = 'gaussian',
                  step_normalize = F,
                  loss = 'mean_squared_error',
                  metrics = 'mean_squared_error',
                  optimizer = optimizer_nadam(learning_rate = .01),
                  validation_split = 0,
                  hidden = NULL,
                  activation.output = 'linear',
                  batch_normalization = F,
                  family_for_init = gaussian,
                  epochs = 100,
                  batch_size = 1000,
                  monitor = 'mean_squared_error',
                  patience = 20,
                  transformation = hirem_transformation_log,
                  filter = function(data){data$payment == 1})

model4e <- hirem::fit(model4e,
                      close = 'close ~ factor(development_year)',
                      payment = 'payment ~ close + factor(development_year)',
                      size = 'size ~ close + development_year_factor')

print(model4e$layers$size$sigma)
simulate_rbns(model4e)

# Let's compare with the GLM, gaussian family:
glm.gaussian <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_glm('size', gaussian(),
            transformation = hirem_transformation_log,
            filter = function(data){data$payment == 1})
glm.gaussian <- hirem::fit(glm.gaussian,
                           close = 'close ~ factor(development_year)',
                           payment = 'payment ~ close + factor(development_year)',
                           size = 'size ~ close + development_year_factor')

# We obtain the same value for sigma:
print(glm.gaussian$layers$size$sigma)

simulate_rbns(glm.gaussian)
# In both case, we see that this distribution choice leads to an RBNS overestimation.

#=========================================================================#
#         Case 4f: GLM + MLP (gamma, 3 hidden layers, autoencoder)        #
#=========================================================================#

init()
model4f <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dnn('size', distribution = 'gamma',
                  ae.hidden = c(40,30,20),
                  ae.activation.hidden = rep('tanh',3),
                  step_log = F,
                  step_normalize = F,
                  loss = gamma_deviance_keras,
                  metrics = metric_gamma_deviance_keras,
                  optimizer = optimizer_nadam(learning_rate = .01),
                  validation_split = 0,
                  hidden = c(20,15,10),
                  activation.output = 'exponential',
                  batch_normalization = F,
                  family_for_init = Gamma(link=log),
                  epochs = 5,
                  batch_size = 1000,
                  monitor = 'gamma_deviance_keras',
                  patience = 20,
                  filter = function(data){data$payment == 1})

model4f <- hirem::fit(model4f,
                      close = 'close ~ factor(development_year)',
                      payment = 'payment ~ close + factor(development_year)',
                      size = 'size ~ close + development_year_factor')

# The shape parameter is (almost) identical to model 1 (glm, gamma log link for size):
print(model4f$layers$size$shape)
print(model4f$layers$size$shape.se)

print(model1$layers$size$shape)
print(model1$layers$size$shape.se)

# Due to the bias regularization, we obtain RBNS simulations close to the true value:
simulate_rbns(model4f)

#=========================================================================#
#         Case 4g: GLM + MLP (gamma, 3 hidden layers, bayesian opt)       #
#=========================================================================#

init()

bounds <- list(
  #ae_hidden_1 = c(10L,15L)
  mlp_hidden_1 = c(10L,30L)
)


model4g <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dnn('size', distribution = 'gamma',
                  bayesOpt = T,
                  bayesOpt_min = T,
                  bayesOpt_iters_n = 1,
                  bayesOpt_bounds = bounds,
                  use_embedding = T,
                  output_dim = 1,
                  hidden = c(20,15,10),
                  nfolds=2,
                  step_log = F,
                  step_normalize = F,
                  verbose = 1,
                  bias_regularization = TRUE,
                  loss = gamma_deviance_keras,
                  metrics = metric_gamma_deviance_keras,
                  optimizer = optimizer_nadam(),
                  validation_split = .3,
                  activation.output = 'exponential',
                  batch_normalization = F,
                  family_for_init = Gamma(link=log),
                  epochs = 10,
                  batch_size = 1000,
                  monitor = 'gamma_deviance_keras',
                  patience = 2,
                  filter = function(data){data$payment == 1})

model4g <- hirem::fit(model4g,
                      weights = weight,
                      weight.var = 'development_year',
                      balance.var = 'development_year',
                      close = 'close ~ factor(development_year)',
                      payment = 'payment ~ close + factor(development_year)',
                      size = 'size ~ close + development_year_factor')

# The shape parameter is (almost) identical to model 1 (glm, gamma log link for size):
print(model4g$layers$size$shape)
print(model4g$layers$size$shape.se)

simulate_rbns(model4g)

#=========================================================================#
#         Case 4h: GLM + MLP (gamma, 3 hidden layers, embedding layers)   #
#=========================================================================#

init()
reserving_data <- reserving_data %>% mutate(X1_factor = factor(X1))
model4h <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dnn('size', distribution = 'gamma', bias_regularization = T,
                  use_embedding = T,
                  output_dim = 1,
                  step_log = F,
                  step_normalize = F,
                  loss = gamma_deviance_keras,
                  metrics = metric_gamma_deviance_keras,
                  optimizer = optimizer_nadam(),
                  validation_split = 0,
                  hidden = c(20,15,10),
                  activation.output = 'exponential',
                  batch_normalization = F,
                  family_for_init = Gamma(link=log),
                  epochs = 5,
                  verbose = 1,
                  batch_size = 1000,
                  monitor = 'gamma_deviance_keras',
                  patience = 20,
                  filter = function(data){data$payment == 1})

model4h <- hirem::fit(model4h,
                      weights = weight,
                      weight.var = 'development_year',
                      balance.var = 'development_year',
                      close = 'close ~ factor(development_year)',
                      payment = 'payment ~ close + factor(development_year)',
                      size = 'size ~ close + development_year_factor')

# The shape parameter is (almost) identical to model 1 (glm, gamma log link for size):
print(model4h$layers$size$shape)
print(model4h$layers$size$shape.se)

# Due to the bias regularization, we obtain RBNS simulations close to the true value:
simulate_rbns(model4h)

#=========================================================================#
#             Case 5: GLM + CANN (gamma) + Bayesian opt                   #
#=========================================================================#

init()

bounds <- list(
  mlp_hidden_1 = c(20L,30L)
)

model5 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_cann('size', distribution = 'gamma', bias_regularization = FALSE,
             bayesOpt = T,
             bayesOpt_min = T,
             bayesOpt_iters_n = 1,
             bayesOpt_bounds = bounds,
             family_for_glm = Gamma(link=log),
             loss = gamma_deviance_keras,
             metrics = metric_gamma_deviance_keras,
             optimizer = optimizer_nadam(),
             hidden = c(20,15,10),
             validation_split = 0,
             activation.output = 'linear',
             activation.output.cann = 'exponential',
             fixed.cann = T,
             monitor = 'gamma_deviance_keras',
             patience = 2,
             epochs = 2,
             batch_size = 1000,
             filter = function(data){data$payment == 1})

model5 <- hirem::fit(model5,
                     weights = weight,
                     weight.var = 'development_year',
                     balance.var = 'development_year',
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year_factor')

print(model5$layers$size$shape)
print(model5$layers$size$shape.se)

simulate_rbns(model5)

#=========================================================================#
#             Case 5b: GLM + CANN (bernoulli)                             #
#=========================================================================#

init()
model5b <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_cann('close', distribution = 'bernoulli',
             family_for_glm = binomial(),
             loss = 'binary_crossentropy',
             metrics = 'binary_crossentropy',
             optimizer = optimizer_nadam(),
             validation_split = 0,
             hidden = c(10,10,10),
             activation.output = 'linear',
             activation.output.cann = 'sigmoid',
             fixed.cann = TRUE,
             monitor = 'binary_crossentropy',
             patience = 20,
             epochs = 100,
             batch_size = 1000,
             verbose = 1) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_glm('size', Gamma(link = log),
            filter = function(data){data$payment == 1})

model5b <- hirem::fit(model5b,
                     close = 'close ~ development_year_factor',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year_factor')

print(model5b$layers$size$shape)
print(model5b$layers$size$shape.se)

simulate_rbns(model5b)

#=========================================================================#
#             Case 5c: GLM + CANN (gaussian)                               #
#=========================================================================#

init()
model5c <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_cann('size', distribution = 'gaussian',
             step_log = TRUE,
             family_for_glm = gaussian,
             loss = 'mean_squared_error',
             metrics = 'mean_squared_error',
             optimizer = optimizer_nadam(learning_rate = .01),
             validation_split = 0,
             hidden = c(20,15,10),
             activation.output = 'linear',
             activation.output.cann = 'linear',
             fixed.cann = FALSE,
             monitor = 'mean_squared_error',
             patience = 20,
             epochs = 100,
             batch_size = 1000,
             filter = function(data){data$payment == 1})

model5c <- hirem::fit(model5c,
                     close = 'close ~ factor(development_year)',
                     payment = 'payment ~ close + factor(development_year)',
                     size = 'size ~ close + development_year_factor')

print(model5c$layers$size$sigma)

#=========================================================================#
#             Case 5d: GLM + CANN (embedding layers)                      #
#=========================================================================#

init()

bounds <- list(
  mlp_hidden_1 = c(25L,30L)
)

reserving_data <- reserving_data %>% mutate(X1_factor = factor(X1))
model5d <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_cann('size', distribution = 'gamma', bias_regularization = T,
             bayesOpt = F, bayesOpt_min = T, bayesOpt_iters_n = 1, bayesOpt_bounds = bounds,
             use_embedding = TRUE,
             formula.glm = 'size ~ close + development_year_factor',
             family_for_glm = Gamma(link=log),
             loss = gamma_deviance_keras,
             metrics = metric_gamma_deviance_keras,
             optimizer = optimizer_nadam(),
             validation_split = 0,
             hidden = c(20,15,10),
             activation.output = 'linear',
             activation.output.cann = 'exponential',
             fixed.cann = TRUE,
             monitor = 'gamma_deviance_keras',
             verbose = 0,
             patience = 2,
             epochs = 5,
             batch_size = 1000,
             filter = function(data){data$payment == 1})

model5d <- hirem::fit(model5d,
                      weights = weight,
                      weight.var = 'development_year',
                      balance.var = 'development_year',
                      close = 'close ~ factor(development_year)',
                      payment = 'payment ~ close + factor(development_year)',
                      size = 'size ~ close + development_year_factor + X1_factor')

print(model5d$layers$size$shape)

simulate_rbns(model5d,30)

#=========================================================================#
#                                 Annex                                   #
#=========================================================================#

### Case A.1: GLM + MLP (h2o) ###
init()
modelA.1 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dnn_h2o('size', distribution = 'gaussian',
                epochs = 1,
                nfolds = 6,
                hidden = c(10,30,50,30,10),
                hidden_dropout_ratios = rep(.01,5),
                activation = 'RectifierWithDropout',
                filter = function(data){data$payment == 1})

modelA.1 <- hirem::fit(modelA.1,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

# Rmk: bias regularization not implemented in layer_dnn_h2o
simulate_rbns(modelA.1)


### Case A.2: GLM + MLP (h2o) ###
init()
modelA.2 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_dnn_h2o('size', distribution = 'gaussian',
                epochs = 10,
                hidden = c(10,10),
                hidden_dropout_ratios = c(0.1,0.1),
                activation = 'TanhWithDropout',
                filter = function(data){data$payment == 1})

modelA.2 <- hirem::fit(modelA.2,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

# Rmk: bias regularization not implemented in layer_dnn_h2o
simulate_rbns(modelA.2)

### Case A.3: GLM + AutoML (h2o) ###
init()
modelA.3 <- hirem(reserving_data) %>%
  split_data(observed = reserving_data %>% dplyr::filter(calendar_year <= 6),
             validation = .7, cv_fold = 6) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_aml_h2o('size', distribution = 'gaussian',
                filter = function(data){data$payment == 1})

modelA.3 <- hirem::fit(modelA.3,
                     close = 'close ~ development_year',
                     payment = 'payment ~ close + development_year',
                     size = 'size ~ close + development_year')

# Rmk: bias regularization not implemented in layer_aml_h2o
simulate_rbns(modelA.3)
