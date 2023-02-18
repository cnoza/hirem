formula_settle_glm <- "settlement ~ type + age.fact + rep.month"
formula_pay_glm <- "payment ~ settlement + devYearMonth + type + rep.month + age.fact"
formula_recov_glm <- "recovery ~ devYearMonth + settlement + type + rep.month + age.fact"
formula_size_pay_glm <- "size.pay ~ type + devYearMonth + age.fact + rep.month + settlement"
formula_size_recov_glm <- "q.size.recov ~ devYearMonth + rep.month + type + age.fact + settlement"

seed_in_error = c()
i=1

gamma_deviance_keras <- function(yobs, yhat) {
  K <- backend()
  2*K$mean((yobs-yhat)/yhat - K$log(yobs/yhat))
}

metric_gamma_deviance_keras <- keras::custom_metric("gamma_deviance_keras", function(yobs, yhat) {
  gamma_deviance_keras(yobs, yhat)
})

for(seed in 9:9) {

  ref_claim <- 1          # currency is 1
  time_unit <- 1/12       # we consider monthly claims development
  set_parameters(ref_claim=ref_claim, time_unit=time_unit)

  years <- 17               # number of occurrence years
  I <- years/time_unit      # number of development periods

  source("~/R-projects/hirem/Examples/wang-wuthrich/Tools/functions simulation.R")
  dest_path <- "" # plot to be saved in ...

  #####################################################################
  #### generate claim data
  #####################################################################

  # generate individual claims: may take 60 seconds
  claims_list <- data.generation(seed=seed, future_info=F)

  # We unset the seed
  set.seed(NULL)

  # save individual output for easier access
  claims <- claims_list[[1]]
  claims <- claims %>% filter(Status == 'Closed')

  paid <- claims_list[[2]]

  # We now build a reserving dataset compatible with hirem

  d0 <- merge(paid, claims, by="Id") %>%
    arrange(Id, EventId)

  d0 <- d0 %>%
    mutate(calendar.year = floor(EventMonth/12)+1,
           rep.year = floor(RepMonth/12)+1,
           dev.year = calendar.year - rep.year + 1,
           size = Paid) %>%
    select(Id, dev.year, size)

  d1 <- expand.grid(Id = unique(d0$Id),
                    dev.year = 1:max(d0$dev.year),
                    size = 0)

  d2 <- union_all(d0,d1)

  d3 <- d2 %>%
    group_by(Id, dev.year) %>%
    dplyr::summarise(total = sum(size))
  colnames(d3) <- c('Id','dev.year', 'size')

  d4 <- merge(d3, claims, by="Id", all.x = T)

  reserving_data <- d4 %>%
    mutate(rep.year = floor(RepMonth/12)+1,
           calendar.year = dev.year + rep.year - 1,
           settlement.year = floor(SetMonth/12)+1,
           payment = as.integer(size>0),
           recovery = as.integer(size<0),
           size.pay = ifelse(size>0,size,0),
           size.recov = ifelse(size<0, -size, 0),
           occ.month = AccMonth,
           occ.date = AccDate,
           occ.weekday = factor(AccWeekday),
           rep.date = RepDate,
           age = Age,
           age.fact = factor(Age),
           settlement = as.integer(calendar.year >= settlement.year),
           open = as.integer(ifelse(calendar.year==settlement.year,1,1-settlement)),
           type = factor(Type),
           rep.month = factor(month(ymd(RepDate)))) %>%
    select(-CumPaid, -Type, -Ultimate, -Status, -PayCount, -AccMonth, -AccDate, -Age, -AccWeekday, -RepDate, RepMonth) %>%
    arrange(Id, dev.year)

  reserving_data <- reserving_data %>% filter(rep.year <= 10 & dev.year <= 10)

  ultimate <- reserving_data %>%
    dplyr::group_by(Id) %>%
    dplyr::summarise(total.size.pay = sum(size.pay))

  reserving_data <- merge(reserving_data, ultimate, by="Id", all.x=T)

  reserving_data <- reserving_data %>%
    mutate(p.size.recov = ifelse(size<0, -size/total.size.pay, 0),
           q.size.recov = ifelse(size<0, log(p.size.recov/(1-p.size.recov)), 0))

  reserving_data$dev.year.fact <- factor(reserving_data$dev.year, levels = 1:max(reserving_data$dev.year))
  reserving_data$rep.year.fact <- factor(reserving_data$rep.year, levels = 1:max(reserving_data$rep.year))

  reserving_data$monthDev12 <- as.character(reserving_data$rep.month)
  reserving_data$monthDev12[reserving_data$dev.year > 2] <- 'dev.year.over.2'
  reserving_data$devYearMonth <- factor(paste(reserving_data$dev.year, reserving_data$monthDev12, sep = '-'))

  observed_data   <- reserving_data %>% filter(calendar.year <= 10)
  prediction_data <- reserving_data %>% filter(calendar.year > 10)

  reported_claims <- observed_data %>%
    dplyr::filter(dev.year == 1) %>%
    group_by(rep.year) %>%
    dplyr::summarise(count = n()) %>%
    pull(count)

  denominator <- tail(rev(cumsum(reported_claims)), -1)
  numerator <- head(cumsum(rev(reported_claims)), -1)
  weight <- c(10^(-6), numerator / denominator)

  names(weight) <- paste0('dev.year',1:10)

  model_cann <-  hirem(reserving_data) %>%
    split_data(observed = function(df) df %>% filter(calendar.year <= 10, open == 1)) %>%
    layer_cann('settlement', distribution = 'bernoulli',
               hidden = c(10,20,10),
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
               hidden = c(10,10,20),
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
    layer_cann('recovery', distribution = 'bernoulli',
               hidden = c(10,10,20),
               formula.glm = formula_recov_glm,
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
    layer_cann('size.pay', distribution = 'gamma',
               hidden = c(30,20,10),
               formula.glm = formula_size_pay_glm,
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
               filter = function(data){data$payment == 1}) %>%
    layer_glm(name = 'q.size.recov', 'family' = gaussian(),
              filter = function(x){x$recovery == 1})

  obj <- model_cann
  data <- model_cann$data_observed

  if(!is.null(obj$weights)) weights.vec <- obj$weights[data[[obj$weight.var]]] else weights.vec <- NULL
  weights.vec.n <- NULL

  for(index in seq_along(obj$layers)) {

    layer <- model_cann$layers[[index]]

    if(layer$method == "cann") {

      data <- data[layer$filter(data), ]
      f.glm <- as.formula(layer$method_options$formula.glm)
      label <- layer$name

      data_recipe.glm <- recipe(f.glm, data=data)
      data_recipe.glm <- data_recipe.glm %>% prep()
      data_baked.glm <- bake(data_recipe.glm, new_data = data)

      model.glm       <- glm(f.glm, data = data_baked.glm, family = layer$method_options$family_for_glm, weights = weights.vec.n)

      data_recipe.glm <- data_recipe.glm %>% step_dummy(all_nominal(), one_hot = FALSE) %>% prep()
      data_baked.glm <- bake(data_recipe.glm, new_data = data)
      x.glm       <- select(data_baked.glm,-as.name(label)) %>% as.matrix()

      length.glm = length(model.glm$coefficients)-1
      length.x.glm = length(dimnames(x.glm)[[2]])

      if(length.glm != length.x.glm) {
        seed_in_error[i] = seed
        cat(sprintf("Seed in error: %s", seed))
        cat(sprintf(" (layer: %s)\n", layer$name))
        i = i+1
        break;
      }

    }

  }

}


# seeds to keep:
# c(1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 25, 28, 29, 32, 33, 34, 35, 37, 38, 39, 41, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 60, 61, 62, 63, 65, 66, 67, 68, 72, 73, 74, 75, 76, 77, 78, 80, 82, 83, 85, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 100, 101, 103, 105, 106, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 119, 120, 122, 123, 124, 127, 129, 131, 132, 134, 137, 139)

# extra seeds: 140, 141, 142, 143

