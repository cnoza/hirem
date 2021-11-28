### Initialization
init <- function() {
  rm(list = ls(envir = .GlobalEnv), envir = .GlobalEnv)
  #install.packages('devtools')
  #devtools::install_github('cnoza/hirem')
  library(hirem)
  source(file='./tests/import/functions.R')
  set.seed(265)
  tensorflow::set_random_seed(265)

  ### Loading data ###
  data("reserving_data")
  reserving_data <<- reserving_data %>%
    dplyr::mutate(development_year_factor = factor(development_year))

  reported_claims <- reserving_data %>%
    dplyr::filter(calendar_year <= 6) %>%
    dplyr::filter(development_year == 1) %>%
    dplyr::group_by(reporting_year) %>%
    dplyr::summarise(count = dplyr::n()) %>%
    dplyr::pull(count)

  denominator <- tail(rev(cumsum(reported_claims)), -1)
  numerator <- head(cumsum(rev(reported_claims)), -1)
  weight <<- c(10^(-6), numerator / denominator)

  names(weight) <- paste0('development_year',1:6)

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

  if(!is.null(model$balance.var)) balance.correction = T

  simul <- simulate(model,
                    nsim = nsim,
                    filter = function(data){dplyr::filter(data,
                                                          development_year <= 6,
                                                          close == 0)},
                    data = reserving_data %>% dplyr::filter(calendar_year == 6),
                    balance.correction = balance.correction)

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
