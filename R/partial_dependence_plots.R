#' Marginal effects of a predictor on a hierarchical model layer.
#'
#' This function can be used to obtain the marginal effects of a predictor on the layer of a hierarchical model.
#' Compatible with layers defined as a GLM, GBM, XGB, DNN or CANN model.
#' The returned data frame can serve as input to draw partial dependence plots (see examples in the package repository).
#'
#' @param object The hierarchical reserving model layer
#' @param data The input data set
#' @param grid A data frame with the values of the predictor for which marginal effects are to be computed
#'
#' @export
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

