
### Functions used for testing purposes

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

gamma_deviance_keras <- function(yobs, yhat) {
  K <- backend()
  2*K$sum((yobs-yhat)/yhat - K$log(yobs/yhat))
}

metric_gamma_deviance_keras <- keras::custom_metric("gamma_deviance_keras", function(yobs, yhat) {
  gamma_deviance_keras(yobs, yhat)
})
