par_dep <- function(object, data, grid) {
  # Initialize a vector to save the effect
  pd_effect <- rep(0, nrow(grid))
  # Iterate over the grid values to calculate the effect
  for (i in seq_len(length(pd_effect))) {
    if(object$method == "glm") {
      pd_effect[i] <-
        data %>%
        dplyr::mutate(!! names(grid) := grid[i, ]) %>%
        predict(object$fit, newdata = ., type = 'response') %>%
        mean()
    }
    else if(object$method == "gbm") {
      pd_effect[i] <-
        data %>%
        dplyr::mutate(!! names(grid) := grid[i, ]) %>%
        predict(object$fit, newdata = ., n.trees = object$fit$n.trees, type = 'response') %>%
        mean()
    }
    else if(object$method == "xgb") {
      data <- data %>% dplyr::mutate(!! names(grid) := grid[i, ])
      f <- as.formula(object$formula)
      label <- as.character(terms(f)[[2]])
      data_baked <- bake(object$data_recipe, new_data = data)
      newdata <- select(data_baked,-as.name(label)) %>% as.matrix()
      newdata.xgb <- xgb.DMatrix(data = newdata, info = list('label' = as.matrix(data[,label])))
      response <- predict(object$fit, ntreelimit = object$fit$niter, newdata = newdata.xgb, type = 'response')
      pd_effect[i] <- mean(response)
    }
    else if(object$method == "dnn") {
      data <- data %>% dplyr::mutate(!! names(grid) := grid[i, ])
      f <- as.formula(object$formula)
      label <- as.character(terms(f)[[2]])

      data_baked <- bake(object$data_recipe, new_data = data)
      if(ncol(data_baked) == 1)
        data_baked <- data_baked %>% mutate(intercept = 1)

      x <- select(data_baked,-as.name(label)) %>% as.matrix()

      def_x <- def_x_mlp(object$method_options$use_embedding,
                         object$method_options$embedding_var,
                         f,
                         data,
                         data_baked,
                         object$data_recipe,
                         label)

      if(!object$method_options$use_embedding) {
        x.inputs <- list(def_x$x)
      }
      else {
        x.inputs <- list(def_x$x_no_fact,def_x$x_fact)
        x.inputs[sapply(x.inputs, is.null)] <- NULL
      }

      if(!is.null(object$method_options$ae.hidden)) {
        x <- data.frame(object$model_en %>% predict(x)) %>% as.matrix()
      }

      if(!object$method_options$bias_regularization) {
        response <- predict(object$fit, x.inputs)
      }
      else {
        Zlearn   <- data.frame(object$zz %>% predict(x.inputs))
        names(Zlearn) <- paste0('X', 1:ncol(Zlearn))
        response <- predict(object$fit, newdata = Zlearn, type = 'response') %>% as.matrix()
      }
      pd_effect[i] <- mean(response)
    }
    else if(object$method == "cann") {
      data <- data %>% dplyr::mutate(!! names(grid) := grid[i, ])
      f <- as.formula(object$formula)
      label <- as.character(terms(f)[[2]])

      if(!is.null(object$method_options$formula.glm))
        f.glm <- as.formula(object$method_options$formula.glm)
      else
        f.glm <- f

      data_baked_for_glm <- bake(object$data_recipe.glm.no_dummy, new_data = data)
      glm.pred <- predict(object$model.glm, newdata = data_baked_for_glm)

      data_baked <- bake(object$data_recipe, new_data = data)
      if(ncol(data_baked) == 1)
        data_baked <- data_baked %>% mutate(intercept = 1)

      data_baked.glm <- bake(object$data_recipe.glm, new_data = data)
      if(ncol(data_baked.glm) == 1)
        data_baked.glm <- data_baked.glm %>% mutate(intercept = 1)

      x     <- select(data_baked,-as.name(label)) %>% as.matrix()
      x.glm <- select(data_baked.glm,-as.name(label)) %>% as.matrix()

      def_x <- def_x(object$method_options$use_embedding,
                     object$method_options$embedding_var,
                     object$method_options$embedding_var.glm,
                     f,
                     f.glm,
                     data,
                     data_baked,
                     data_baked.glm,
                     object$data_recipe,
                     object$data_recipe.glm,
                     label)

      if(!object$method_options$use_embedding) {
        x.inputs <- list(def_x$x,def_x$x.glm)
      }
      else {
        x.inputs <- list(def_x$x_no_fact.glm,def_x$x_fact.glm,def_x$x_no_fact,def_x$x_fact)
        x.inputs[sapply(x.inputs, is.null)] <- NULL
      }

      if(!object$method_options$bias_regularization) {
        response <- predict(object$fit, x.inputs)
      }
      else {
        Zlearn   <- data.frame(object$zz %>% predict(x.inputs))
        names(Zlearn) <- paste0('X', 1:ncol(Zlearn))
        Zlearn$glm.pred <- glm.pred
        response <- predict(object$fit, newdata = Zlearn, type = 'response') %>% as.matrix()
      }
      pd_effect[i] <- mean(response)
    }
  }
  return(pd_effect)
}
#'

df.train <- reserving_data %>% dplyr::filter(calendar.year <= 9, open == 1)
#df.train_sample <- df.train[sample(seq_len(nrow(df.train)), size = 10000), ]

# Partial dependence plots for "type" covariate

grid_type_init <- data.frame('type' = as.factor(levels(df.train$type)))

# Response variable: settlement

grid_type_settle <- grid_type_init %>%
  dplyr::mutate(glm = model_glm$layers$settlement %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(gbm = model_gbm$layers$settlement %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(xgb = model_xgb$layers$settlement %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(dnn = model_dnn$layers$settlement %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(cann = model_cann$layers$settlement %>% par_dep(data = df.train, grid = grid_type_init))

# PDPlot

pd_type_settle <- grid_type_settle %>% reshape2::melt(id.vars = 'type',
                              value.name = 'pd_settlement',
                              variable.name = 'method') %>%
  ggplot(aes(x = type, y = pd_settlement)) + theme_bw() +
  geom_line(aes(group = method, colour = method))

pd_type_settle

# Response variable: payment

grid_type_payment <- grid_type_init %>%
  dplyr::mutate(glm = model_glm$layers$payment %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(gbm = model_gbm$layers$payment %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(xgb = model_xgb$layers$payment %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(dnn = model_dnn$layers$payment %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(cann = model_cann$layers$payment %>% par_dep(data = df.train, grid = grid_type_init))

# PDPlot

pd_type_payment <- grid_type_payment %>% reshape2::melt(id.vars = 'type',
                             value.name = 'pd_payment',
                             variable.name = 'method') %>%
  ggplot(aes(x = type, y = pd_payment)) + theme_bw() +
  geom_line(aes(group = method, colour = method))

pd_type_payment

# Response variable: size

grid_type_size <- grid_type_init %>%
  dplyr::mutate(glm = model_glm$layers$size %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(gbm = model_gbm$layers$size %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(xgb = model_xgb$layers$size %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(dnn = model_dnn$layers$size %>% par_dep(data = df.train, grid = grid_type_init)) %>%
  dplyr::mutate(cann = model_cann$layers$size %>% par_dep(data = df.train, grid = grid_type_init))

# PDPlot

pd_type_size <- grid_type_size %>% reshape2::melt(id.vars = 'type',
                             value.name = 'pd_size',
                             variable.name = 'method') %>%
  ggplot(aes(x = type, y = pd_size)) + theme_bw() +
  geom_line(aes(group = method, colour = method))

pd_type_size

library(gridExtra)
grid.arrange(pd_type_settle,pd_type_payment,pd_type_size,ncol=3)
