library(gbm)

#hyper_grid <- hyper_grid[1,]

data <- observed_data %>% filter(calendar.year <= 9, open == 1)

reported_claims <- observed_data %>%
  dplyr::filter(dev.year == 1) %>%
  group_by(rep.year) %>%
  dplyr::summarise(count = n()) %>%
  pull(count)

denominator <- tail(rev(cumsum(reported_claims)), -1)
numerator <- head(cumsum(rev(reported_claims)), -1)
weight <- c(10^(-6), numerator / denominator)

names(weight) <- paste0('dev.year',1:9)

weights.vec <- weight[data[['dev.year']]]

covariates_gbm <- c('type', 'dev.year.fact', 'rep.month', 'rep.year.fact', 'rep.delay', 'calendar.year')

formula_settl <- paste0('settlement ~ 1 + ', paste0(covariates_gbm, collapse = ' + '))
formula_pay <- paste0('payment ~ 1 + ', paste0(c(covariates_gbm, 'settlement'), collapse = ' + '))
formula_recov <- paste0('recovery ~ 1 + ', paste0(c(covariates_gbm, 'settlement'), collapse = ' + '))
formula_size_pay <- paste0('size.pay ~ 1 + ', paste0(c(covariates_gbm,'settlement'), collapse = ' + '))
formula_size_recov <- paste0('q.size.recov ~ 1 + ', paste0(c(covariates_gbm,'settlement'), collapse = ' + '))

distribution_bern <- 'bernoulli'
distribution_gamma <- 'gamma'

hyper_grid <- expand.grid('shrinkage' = c(0.05),
                          'interaction.depth' = c(1:3))

iter <- c()
cv_error <- c()

for(i in seq_len(nrow(hyper_grid))){

  mod_gbm <- gbm(as.formula(formula_size_recov),
                   data = data,
                   distribution = 'gamma',
                   n.trees = 700,
                   cv.folds = 5,
                   interaction.depth = hyper_grid$interaction.depth[i],
                   shrinkage = hyper_grid$shrinkage[i],
                   n.minobsinnode = 100,
                   bag.fraction = 0.75,
                   weights = weights.vec,
                   keep.data = TRUE)

  iter[i] <- gbm.perf(mod_gbm, plot.it = TRUE)

  #iter[i] <- which.min(mod_gbm$oobag.improve[is.finite(mod_gbm$oobag.improve)])
  cv_error[i] <- mod_gbm$cv.error[[iter[i]]]
}

hyper_grid$iter <- iter
hyper_grid$cv_error <- cv_error
