par_dep <- function(object, data, grid) {
  # Initialize a vector to save the effect
  pd_effect <- rep(0, nrow(grid))
  # Iterate over the grid values to calculate the effect
  for (i in seq_len(length(pd_effect))) {
    if(class(object$fit) == "gbm") {
      pd_effect[i] <-
        data %>%
        dplyr::mutate(!! names(grid) := grid[i, ]) %>%
        predict(object$fit, newdata = ., n.trees = object$fit$n.trees, type = 'response') %>%
        mean()
    }
    else if(class(object$fit) == "xgb.Booster") {
      data <- data %>% dplyr::mutate(!! names(grid) := grid[i, ])
      f <- as.formula(object$formula)
      label <- as.character(terms(f)[[2]])
      data_baked <- bake(object$data_recipe, new_data = data)
      newdata <- select(data_baked,-as.name(label)) %>% as.matrix()
      newdata.xgb <- xgb.DMatrix(data = newdata, info = list('label' = as.matrix(data[,label])))
      response <- predict(object$fit, ntreelimit = object$fit$niter, newdata = newdata.xgb, type = 'response')
      pd_effect[i] <- mean(response)
    }
  }
  return(pd_effect)
}
#'

df.train <- reserving_data %>% dplyr::filter(calendar.year <= 9, open == 1)
df.train_sample <- df.train[sample(seq_len(nrow(df.train)), size = 10000), ]

grid_type <- data.frame('type' = as.factor(levels(df.train$type)))
grid_type_init <- grid_type

# Partial dependence plots for settlement

# GBM, XGB

grid_type <- grid_type_init %>%
  dplyr::mutate(gbm = model_gbm$layers$settlement %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(xgb = model_xgb$layers$settlement %>% par_dep(data = df.train, grid = grid_type_init))


# PDPlot settlement

grid_type %>% reshape2::melt(id.vars = 'type',
                              value.name = 'pd_settlement',
                              variable.name = 'method') %>%
  ggplot(aes(x = type, y = pd_settlement)) + theme_bw() +
  geom_line(aes(group = method, colour = method))

