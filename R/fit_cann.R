

#' @importFrom data.table transpose
#' @importFrom caret createFolds
#' @export
fit.layer_cann <- function(layer, obj, formula, training = FALSE, fold = NULL) {
  cat(sprintf("Fitting layer_cann for %s...\n", layer$name))

  layer$formula <- formula

  data <- obj$data_training
  if(!training) {
    data <- obj$data_observed
  }

  if(!is.null(fold)) {
    data <- data %>% filter(cv_fold != fold)
  }

  data <- data[layer$filter(data), ]

  f     <- as.formula(formula)
  label <- as.character(terms(f)[[2]])

  if(!is.null(layer$method_options$formula.glm))
    f.glm <- as.formula(layer$method_options$formula.glm)
  else
    f.glm <- f

  if(!is.null(layer$transformation)) {
    data[,label] <- layer$transformation$transform(data[,label])
  }

  if(!is.null(obj$weights)) weights.vec <- obj$weights[data[[obj$weight.var]]] else weights.vec <- NULL

  if(!is.null(weights.vec)) weights.vec.n <- weights.vec*length(weights.vec)/sum(weights.vec)
  else weights.vec.n <- NULL

  # for testing purposes...
  weights.vec.n <- NULL

  data_recipe     <- recipe(f, data=data)
  data_recipe.glm <- recipe(f.glm, data=data)

  if(layer$method_options$step_log) {
    data_recipe     <- data_recipe %>% step_log(as.name(label))
    data_recipe.glm <- data_recipe.glm %>% step_log(as.name(label))
  }
  if(layer$method_options$step_normalize) {
    data_recipe     <- data_recipe %>% step_normalize(all_numeric(), -all_outcomes())
    data_recipe.glm <- data_recipe.glm %>% step_normalize(all_numeric(), -all_outcomes())
  }

  data_recipe     <- data_recipe %>% prep()
  data_recipe.glm <- data_recipe.glm %>% prep()
  layer$data_recipe.glm.no_dummy <- data_recipe.glm

  data_baked.glm <- bake(data_recipe.glm, new_data = data)

  if(ncol(data_baked.glm) == 1)
    data_baked.glm <- data_baked.glm %>% mutate(intercept = 1)

  model.glm       <- glm(f.glm, data = data_baked.glm, family = layer$method_options$family_for_glm, weights = weights.vec.n)
  layer$model.glm <- model.glm

  if(!layer$method_options$use_embedding) {
    data_recipe.glm <- data_recipe.glm %>% step_dummy(all_nominal(), one_hot = FALSE) %>% prep()
    data_recipe     <- data_recipe %>% step_dummy(all_nominal(), one_hot = TRUE) %>% prep()
  }

  data_baked.glm <- bake(data_recipe.glm, new_data = data)
  data_baked <- bake(data_recipe, new_data = data)
  nrows <- nrow(data_baked)

  if(ncol(data_baked) == 1)
    data_baked <- data_baked %>% mutate(intercept = 1)

  if(ncol(data_baked.glm) == 1)
    data_baked.glm <- data_baked.glm %>% mutate(intercept = 1)

  layer$data_recipe.glm <- data_recipe.glm
  layer$data_recipe     <- data_recipe

  def_x <- def_x(layer$method_options$use_embedding,
                 layer$method_options$embedding_var,
                 layer$method_options$embedding_var.glm,
                 f,
                 f.glm,
                 data,
                 data_baked,
                 data_baked.glm,
                 data_recipe,
                 data_recipe.glm,
                 label)

  layer$x     <- def_x$x
  layer$x.glm <- def_x$x.glm

  if(layer$method_options$use_embedding) {
    layer$x_fact         <- def_x$x_fact
    layer$x_no_fact      <- def_x$x_no_fact
    layer$x_fact.glm     <- def_x$x_fact.glm
    layer$x_no_fact.glm  <- def_x$x_no_fact.glm
    layer$fact_var       <- def_x$fact_var
    layer$no_fact_var    <- def_x$no_fact_var
    layer$fact_var.glm   <- def_x$fact_var.glm
    layer$no_fact_var.glm  <- def_x$no_fact_var.glm
  }

  y       <- data_baked %>% pull(as.name(label))
  layer$y <- y

  if(layer$method_options$gridsearch_cv) {

    Folds <- list()
    names <- c()
    folds <- caret::createFolds(y = y, k = layer$method_options$nfolds, list = F)
    data_baked$folds <- folds

    for(i in 1:layer$method_options$nfolds) {
      #Folds[[i]] <- as.integer(seq(i,nrow(x),by = layer$method_options$nfolds))
      Folds[[i]] <- which(data_baked$folds == i)
      names[i] <- paste0('Fold',i)
    }
    names(Folds) <- names

    if(!is.null(layer$method_options$hyper_grid)) {
      hyper_grid <- layer$method_options$hyper_grid
      if('dnn_hidden_1' %in% names(hyper_grid)) {
        if('dnn_hidden_2' %in% names(hyper_grid)) {
          if('dnn_hidden_3' %in% names(hyper_grid)) {
            hyper_grid <- hyper_grid %>%
              filter(!(dnn_hidden_1 == 0 & ((dnn_hidden_2 != 0) | (dnn_hidden_3 != 0)))) %>%
              filter(!(dnn_hidden_2 == 0 & dnn_hidden_3 != 0))
          }
          else {
            hyper_grid <- hyper_grid %>%
              filter(!(dnn_hidden_1 == 0 & dnn_hidden_2 != 0))
          }
        }
      }
    }
    else {
      hyper_grid <- expand.grid(
        dnn_hidden_1 = seq(from = 10, to = 60, by = 10)
        , dnn_hidden_2 = seq(from = 10, to = 60, by = 10)
        , dnn_hidden_3 = seq(from = 10, to = 60, by = 10)
      )
    }

    if(layer$method_options$random_trials > 0) {
      i <- sample(1:dim(hyper_grid)[1],layer$method_options$random_trials)
      hyper_grid <- hyper_grid[i,]
    }

    cat('_ Initializing the weights...\n')

    init_weights <- list()

    for(j in seq_len(nrow(hyper_grid))) {

      # We only need to do this for 1 fold
      k <- 1

      if(!layer$method_options$use_embedding) {
        x.val         <- layer$x[Folds[[k]],] %>% as.matrix()
        x.glm.val     <- layer$x.glm[Folds[[k]],] %>% as.matrix()
      }
      else {

        x_fact.val <- NULL
        if(length(layer$fact_var)>0) {
          x_fact.val <- list()
          for(i in 1:length(layer$fact_var)) {
            x_fact.val[[i]] <- layer$x_fact[[i]][Folds[[k]]] %>% as.integer()
          }
        }

        x_no_fact.val <- NULL
        if(length(layer$no_fact_var)>0)
          x_no_fact.val  <- layer$x_no_fact[Folds[[k]],] %>% as.matrix()

        x_fact.glm.val <- NULL
        if(length(layer$fact_var.glm)>0) {
          x_fact.glm.val <- list()
          for(i in 1:length(layer$fact_var.glm)) {
            x_fact.glm.val[[i]] <- layer$x_fact.glm[[i]][Folds[[k]]] %>% as.integer()
          }
        }

        x_no_fact.glm.val <- NULL
        if(length(layer$no_fact_var.glm)>0)
          x_no_fact.glm.val <- layer$x_no_fact.glm[Folds[[k]],] %>% as.matrix()

      }

      y.val        <- layer$y[Folds[[k]]]
      sample.w.val <- weights.vec.n[Folds[[k]]]

      if(layer$method_options$nfolds==1) {

        if(!layer$method_options$use_embedding) {
          x     <- x.val
          x.glm <- x.glm.val
        }
        else {
          x_fact        <- x_fact.val
          x_no_fact     <- x_no_fact.val
          x_fact.glm    <- x_fact.glm.val
          x_no_fact.glm <- x_no_fact.glm.val
        }

        y <- y.val
        sample.w <- sample.w.val

      }
      else {

        if(!layer$method_options$use_embedding) {
          x      <- layer$x[-Folds[[k]],]
          x.glm  <- layer$x.glm[-Folds[[k]],]
        }
        else {
          x_fact <- NULL
          if(length(layer$fact_var)>0) {
            x_fact <- list()
            for(i in 1:length(layer$fact_var)) {
              x_fact[[i]] <- layer$x_fact[[i]][-Folds[[k]]] %>% as.integer()
            }
          }
          x_no_fact <- NULL
          if(length(layer$no_fact_var)>0)
            x_no_fact <- layer$x_no_fact[-Folds[[k]],] %>% as.matrix()

          x_fact.glm <- NULL
          if(length(layer$fact_var.glm)>0) {
            x_fact.glm <- list()
            for(i in 1:length(layer$fact_var.glm)) {
              x_fact.glm[[i]] <- layer$x_fact.glm[[i]][-Folds[[k]]] %>% as.integer()
            }
          }
          x_no_fact.glm <- NULL
          if(length(layer$no_fact_var.glm)>0)
            x_no_fact.glm <- layer$x_no_fact.glm[-Folds[[k]],] %>% as.matrix()
        }

        y <- layer$y[-Folds[[k]]]
        sample.w <- weights.vec.n[-Folds[[k]]]

      }

      def_inputs <- def_inputs(use_embedding=layer$method_options$use_embedding,
                               x=x,
                               model.glm=model.glm,
                               no_fact_var=layer$no_fact_var,
                               no_fact_var.glm=layer$no_fact_var.glm,
                               fact_var.glm=layer$fact_var.glm,
                               fact_var=layer$fact_var,
                               x_fact=x_fact,
                               x_fact.glm=x_fact.glm)

      # GLM Neural network

      if(!layer$method_options$use_embedding) {
        GLMNetwork.tmp <- def_inputs$inputs.glm %>%
          layer_dense(units=1, activation='linear', name='output_layer_glm', trainable=FALSE,
                      weights=list(array(model.glm$coefficients[2:length(model.glm$coefficients)], dim=c(length(model.glm$coefficients)-1,1)),
                                   array(model.glm$coefficients[1],dim=c(1))))
      }
      else {

        coef_no_fact.glm <- NULL
        if(length(def_inputs$beta.no_fact_var.glm) > 1)
          coef_no_fact.glm <- def_inputs$beta.no_fact_var.glm[2:length(def_inputs$beta.no_fact_var.glm)]

        coef_fact_var.glm <- NULL
        if(length(layer$fact_var.glm)>0)
          coef_fact_var.glm <- rep(1,length(layer$fact_var.glm))

        GLMNetwork.tmp <- def_inputs$inputs.glm %>%
          layer_dense(units=1, activation='linear', name='output_layer_glm', trainable=FALSE,
                      weights=list(array(c(coef_no_fact.glm,coef_fact_var.glm),
                                         dim=c(length(layer$fact_var.glm)+length(def_inputs$beta.no_fact_var.glm)-1,1)),
                                   array(def_inputs$beta.no_fact_var.glm[1],dim=c(1))))
      }

      # Hyperparameters

      dnn_hidden <- c()

      if(!is.null(layer$method_options$hidden))
        dnn_hidden <- layer$method_options$hidden

      if(!is.null(hyper_grid$dnn_hidden_1) && hyper_grid$dnn_hidden_1[j] > 0) dnn_hidden[1] <- hyper_grid$dnn_hidden_1[j]
      if(!is.null(hyper_grid$dnn_hidden_2) && hyper_grid$dnn_hidden_2[j] > 0) dnn_hidden[2] <- hyper_grid$dnn_hidden_2[j]
      if(!is.null(hyper_grid$dnn_hidden_3) && hyper_grid$dnn_hidden_3[j] > 0) dnn_hidden[3] <- hyper_grid$dnn_hidden_3[j]

      if(length(dnn_hidden)==0) dnn_hidden <- NULL

      dnn_dropout.hidden <- c()

      if(!is.null(layer$method_options$dropout.hidden))
        dnn_dropout.hidden <- layer$method_options$dropout.hidden
      else {
        if(!is.null(dnn_hidden))
          dnn_dropout.hidden <- rep(0,length(dnn_hidden))
      }

      if(!is.null(hyper_grid$dnn_dropout.hidden_1) && hyper_grid$dnn_dropout.hidden_1[j] > 0) dnn_dropout.hidden[1] <- hyper_grid$dnn_dropout.hidden_1[j]
      if(!is.null(hyper_grid$dnn_dropout.hidden_2) && hyper_grid$dnn_dropout.hidden_2[j] > 0) dnn_dropout.hidden[2] <- hyper_grid$dnn_dropout.hidden_2[j]
      if(!is.null(hyper_grid$dnn_dropout.hidden_3) && hyper_grid$dnn_dropout.hidden_3[j] > 0) dnn_dropout.hidden[3] <- hyper_grid$dnn_dropout.hidden_3[j]

      if(length(dnn_dropout.hidden)==0) dnn_dropout.hidden <- NULL

      # Neural network

      NNetwork.tmp <- def_NN_arch(def_inputs$inputs,
                                  layer$method_options$batch_normalization,
                                  dnn_hidden,
                                  layer$method_options$activation.hidden,
                                  dnn_dropout.hidden,
                                  layer$method_options$activation.output,
                                  layer$method_options$use_bias)

      # CANN

      CANNoutput.tmp <- list(GLMNetwork.tmp, NNetwork.tmp) %>% layer_add() %>%
        layer_dense(units = 1, activation = layer$method_options$activation.output.cann, trainable = !layer$method_options$fixed.cann,
                    weights = switch(layer$method_options$fixed.cann + 1,NULL,list(array(c(1), dim=c(1,1)),
                                                                                   array(0, dim=c(1)))),
                    name = 'output_layer_CANN')

      if(!layer$method_options$use_embedding)
        CANN.tmp <- keras_model(inputs = c(def_inputs$inputs,
                                           def_inputs$inputs.glm),
                                outputs = c(CANNoutput.tmp), name = 'CANN.tmp')
      else
        CANN.tmp <- keras_model(inputs = c(def_inputs$inputs_no_fact.glm,
                                           def_inputs$input_layer_emb.glm,
                                           def_inputs$inputs_no_fact,
                                           def_inputs$input_layer_emb),
                                outputs = c(CANNoutput.tmp), name = 'CANN.tmp')

      CANN.tmp %>% compile(
        loss = layer$method_options$loss,
        optimizer = layer$method_options$optimizer,
        metrics = layer$method_options$metrics
      )

      init_weights[[j]] <- keras::get_weights(CANN.tmp)

    }

    cat('_ Starting hypergrid search...\n')

    earlystopping <- callback_early_stopping(
      monitor = layer$method_options$monitor,
      patience = layer$method_options$patience,
      restore_best_weights = T)

    best_score <- ifelse(layer$method_options$gridsearch_cv.min, 10^6, -10^6)

    mean_scores <- c()

    for(j in seq_len(nrow(hyper_grid))) {

      cat(sprintf("_ Now testing hyper_grid row with "))
      cat(sprintf("%s=%s",names(hyper_grid),hyper_grid[j,]))
      cat('\n')

      score <- c()

      for(k in 1:layer$method_options$nfolds) {

        if(!layer$method_options$use_embedding) {
          x.val         <- layer$x[Folds[[k]],] %>% as.matrix()
          x.glm.val     <- layer$x.glm[Folds[[k]],] %>% as.matrix()
        }
        else {

          x_fact.val <- NULL
          if(length(layer$fact_var)>0) {
            x_fact.val <- list()
            for(i in 1:length(layer$fact_var)) {
              x_fact.val[[i]] <- layer$x_fact[[i]][Folds[[k]]] %>% as.integer()
            }
          }

          x_no_fact.val <- NULL
          if(length(layer$no_fact_var)>0)
            x_no_fact.val  <- layer$x_no_fact[Folds[[k]],] %>% as.matrix()

          x_fact.glm.val <- NULL
          if(length(layer$fact_var.glm)>0) {
            x_fact.glm.val <- list()
            for(i in 1:length(layer$fact_var.glm)) {
              x_fact.glm.val[[i]] <- layer$x_fact.glm[[i]][Folds[[k]]] %>% as.integer()
            }
          }

          x_no_fact.glm.val <- NULL
          if(length(layer$no_fact_var.glm)>0)
            x_no_fact.glm.val <- layer$x_no_fact.glm[Folds[[k]],] %>% as.matrix()

        }

        y.val        <- layer$y[Folds[[k]]]
        sample.w.val <- weights.vec.n[Folds[[k]]]

        if(layer$method_options$nfolds==1) {

          if(!layer$method_options$use_embedding) {
            x     <- x.val
            x.glm <- x.glm.val
          }
          else {
            x_fact        <- x_fact.val
            x_no_fact     <- x_no_fact.val
            x_fact.glm    <- x_fact.glm.val
            x_no_fact.glm <- x_no_fact.glm.val
          }

          y <- y.val
          sample.w <- sample.w.val

        }
        else {

          if(!layer$method_options$use_embedding) {
            x      <- layer$x[-Folds[[k]],]
            x.glm  <- layer$x.glm[-Folds[[k]],]
          }
          else {
            x_fact <- NULL
            if(length(layer$fact_var)>0) {
              x_fact <- list()
              for(i in 1:length(layer$fact_var)) {
                x_fact[[i]] <- layer$x_fact[[i]][-Folds[[k]]] %>% as.integer()
              }
            }
            x_no_fact <- NULL
            if(length(layer$no_fact_var)>0)
              x_no_fact <- layer$x_no_fact[-Folds[[k]],] %>% as.matrix()

            x_fact.glm <- NULL
            if(length(layer$fact_var.glm)>0) {
              x_fact.glm <- list()
              for(i in 1:length(layer$fact_var.glm)) {
                x_fact.glm[[i]] <- layer$x_fact.glm[[i]][-Folds[[k]]] %>% as.integer()
              }
            }
            x_no_fact.glm <- NULL
            if(length(layer$no_fact_var.glm)>0)
              x_no_fact.glm <- layer$x_no_fact.glm[-Folds[[k]],] %>% as.matrix()
          }

          y <- layer$y[-Folds[[k]]]
          sample.w <- weights.vec.n[-Folds[[k]]]

        }

        def_inputs <- def_inputs(use_embedding=layer$method_options$use_embedding,
                                 x=x,
                                 model.glm=model.glm,
                                 no_fact_var=layer$no_fact_var,
                                 no_fact_var.glm=layer$no_fact_var.glm,
                                 fact_var.glm=layer$fact_var.glm,
                                 fact_var=layer$fact_var,
                                 x_fact=x_fact,
                                 x_fact.glm=x_fact.glm)

        # GLM Neural network

        if(!layer$method_options$use_embedding) {
          GLMNetwork.tmp <- def_inputs$inputs.glm %>%
            layer_dense(units=1, activation='linear', name='output_layer_glm', trainable=FALSE,
                        weights=list(array(model.glm$coefficients[2:length(model.glm$coefficients)], dim=c(length(model.glm$coefficients)-1,1)),
                                     array(model.glm$coefficients[1],dim=c(1))))
        }
        else {

          coef_no_fact.glm <- NULL
          if(length(def_inputs$beta.no_fact_var.glm) > 1)
            coef_no_fact.glm <- def_inputs$beta.no_fact_var.glm[2:length(def_inputs$beta.no_fact_var.glm)]

          coef_fact_var.glm <- NULL
          if(length(layer$fact_var.glm)>0)
            coef_fact_var.glm <- rep(1,length(layer$fact_var.glm))

          GLMNetwork.tmp <- def_inputs$inputs.glm %>%
            layer_dense(units=1, activation='linear', name='output_layer_glm', trainable=FALSE,
                        weights=list(array(c(coef_no_fact.glm,coef_fact_var.glm),
                                           dim=c(length(layer$fact_var.glm)+length(def_inputs$beta.no_fact_var.glm)-1,1)),
                                     array(def_inputs$beta.no_fact_var.glm[1],dim=c(1))))
        }

        # Hyperparameters

        dnn_hidden <- c()

        if(!is.null(layer$method_options$hidden))
          dnn_hidden <- layer$method_options$hidden

        if(!is.null(hyper_grid$dnn_hidden_1) && hyper_grid$dnn_hidden_1[j] > 0) dnn_hidden[1] <- hyper_grid$dnn_hidden_1[j]
        if(!is.null(hyper_grid$dnn_hidden_2) && hyper_grid$dnn_hidden_2[j] > 0) dnn_hidden[2] <- hyper_grid$dnn_hidden_2[j]
        if(!is.null(hyper_grid$dnn_hidden_3) && hyper_grid$dnn_hidden_3[j] > 0) dnn_hidden[3] <- hyper_grid$dnn_hidden_3[j]

        if(length(dnn_hidden)==0) dnn_hidden <- NULL

        dnn_dropout.hidden <- c()

        if(!is.null(layer$method_options$dropout.hidden))
          dnn_dropout.hidden <- layer$method_options$dropout.hidden
        else {
          if(!is.null(dnn_hidden))
            dnn_dropout.hidden <- rep(0,length(dnn_hidden))
        }

        if(!is.null(hyper_grid$dnn_dropout.hidden_1) && hyper_grid$dnn_dropout.hidden_1[j] > 0) dnn_dropout.hidden[1] <- hyper_grid$dnn_dropout.hidden_1[j]
        if(!is.null(hyper_grid$dnn_dropout.hidden_2) && hyper_grid$dnn_dropout.hidden_2[j] > 0) dnn_dropout.hidden[2] <- hyper_grid$dnn_dropout.hidden_2[j]
        if(!is.null(hyper_grid$dnn_dropout.hidden_3) && hyper_grid$dnn_dropout.hidden_3[j] > 0) dnn_dropout.hidden[3] <- hyper_grid$dnn_dropout.hidden_3[j]

        if(length(dnn_dropout.hidden)==0) dnn_dropout.hidden <- NULL

        # Neural network

        NNetwork.tmp <- def_NN_arch(def_inputs$inputs,
                                    layer$method_options$batch_normalization,
                                    dnn_hidden,
                                    layer$method_options$activation.hidden,
                                    dnn_dropout.hidden,
                                    layer$method_options$activation.output,
                                    layer$method_options$use_bias)

        # CANN

        CANNoutput.tmp <- list(GLMNetwork.tmp, NNetwork.tmp) %>% layer_add() %>%
          layer_dense(units = 1, activation = layer$method_options$activation.output.cann, trainable = !layer$method_options$fixed.cann,
                      weights = switch(layer$method_options$fixed.cann + 1,NULL,list(array(c(1), dim=c(1,1)),
                                                                                     array(0, dim=c(1)))),
                      name = 'output_layer_CANN')

        if(!layer$method_options$use_embedding)
          CANN.tmp <- keras_model(inputs = c(def_inputs$inputs,
                                             def_inputs$inputs.glm),
                                  outputs = c(CANNoutput.tmp), name = 'CANN.tmp')
        else
          CANN.tmp <- keras_model(inputs = c(def_inputs$inputs_no_fact.glm,
                                             def_inputs$input_layer_emb.glm,
                                             def_inputs$inputs_no_fact,
                                             def_inputs$input_layer_emb),
                                  outputs = c(CANNoutput.tmp), name = 'CANN.tmp')

        CANN.tmp %>% compile(
          loss = layer$method_options$loss,
          optimizer = layer$method_options$optimizer,
          metrics = layer$method_options$metrics
        )

        if(layer$method_options$verbose == 1) print(summary(CANN.tmp))

        keras::set_weights(CANN.tmp, init_weights[[j]])

        # earlystopping <- callback_early_stopping(
        #   monitor = layer$method_options$monitor,
        #   patience = layer$method_options$patience,
        #   restore_best_weights = T)

        if(!layer$method_options$use_embedding) {
          x.inputs <- list(x,x.glm)
          x.inputs.val <- list(x.val,x.glm.val)
        }
        else {
          x.inputs <- list(x_no_fact.glm,x_fact.glm,x_no_fact,x_fact)
          x.inputs[sapply(x.inputs, is.null)] <- NULL
          x.inputs.val <- list(x_no_fact.glm.val,x_fact.glm.val,x_no_fact.val,x_fact.val)
          x.inputs.val[sapply(x.inputs.val, is.null)] <- NULL
        }

        if(layer$method_options$nfolds > 1) {
          history.tmp <- CANN.tmp %>%
            keras::fit(x=x.inputs, y=y, sample_weight = sample.w, epochs = layer$method_options$epochs,
                       batch_size = layer$method_options$batch_size,
                       #validation_split = layer$method_options$validation_split,
                       validation_data = list(x.inputs.val,y.val,sample.w.val),
                       callbacks = list(earlystopping),
                       shuffle = layer$method_options$shuffle,
                       verbose = layer$method_options$verbose)
        }
        else {
          history.tmp <- CANN.tmp %>%
            keras::fit(x=x.inputs, y=y, sample_weight = sample.w, epochs = layer$method_options$epochs,
                       batch_size = layer$method_options$batch_size,
                       validation_split = layer$method_options$validation_split,
                       #validation_data = list(x.inputs.val,y.val,sample.w.val),
                       callbacks = list(earlystopping),
                       shuffle = layer$method_options$shuffle,
                       verbose = layer$method_options$verbose)
        }

        score[k] <- ifelse(layer$method_options$gridsearch_cv.min, min(history.tmp$metrics[[4]]), max(history.tmp$metrics[[4]]))

        cat(sprintf('_ Score for fold %s is %s\n',k,score[k]))

      }

      mean_score <- mean(score)

      mean_scores[j] <- mean_score

      cat(sprintf('_ Mean score for hyper_grid row %s is %s\n',j,mean_score))

      if(ifelse(layer$method_options$gridsearch_cv.min, mean_score < best_score, mean_score > best_score)) {

        best_score <- mean_score
        best_j <- j

        cat(sprintf("_ Best score so far is %s for hyper_grid row with ", best_score))
        cat(sprintf("%s=%s",names(hyper_grid),hyper_grid[best_j,]))
        cat('\n')

        if(is.null(layer$method_options$hidden) &
           ( (!is.null(hyper_grid$dnn_hidden_1) && hyper_grid$dnn_hidden_1[j] > 0) |
             (!is.null(hyper_grid$dnn_hidden_2) && hyper_grid$dnn_hidden_2[j] > 0) |
             (!is.null(hyper_grid$dnn_hidden_3) && hyper_grid$dnn_hidden_3[j] > 0)
           )
        ) {
          layer$method_options$hidden <- c()
        }

        if(!is.null(hyper_grid$dnn_hidden_1) && hyper_grid$dnn_hidden_1[j] > 0) layer$method_options$hidden[1] <- hyper_grid$dnn_hidden_1[j]
        if(!is.null(hyper_grid$dnn_hidden_2) && hyper_grid$dnn_hidden_2[j] > 0) layer$method_options$hidden[2] <- hyper_grid$dnn_hidden_2[j]
        if(!is.null(hyper_grid$dnn_hidden_3) && hyper_grid$dnn_hidden_3[j] > 0) layer$method_options$hidden[3] <- hyper_grid$dnn_hidden_3[j]

        if(is.null(layer$method_options$dropout.hidden) &
           ( (!is.null(hyper_grid$dnn_dropout.hidden_1) && hyper_grid$dnn_dropout.hidden_1[j] > 0) |
             (!is.null(hyper_grid$dnn_dropout.hidden_2) && hyper_grid$dnn_dropout.hidden_2[j] > 0) |
             (!is.null(hyper_grid$dnn_dropout.hidden_3) && hyper_grid$dnn_dropout.hidden_3[j] > 0)
           )
        )
          layer$method_options$dropout.hidden <- c()

        if(!is.null(hyper_grid$dnn_dropout.hidden_1) && hyper_grid$dnn_dropout.hidden_1[j] > 0) layer$method_options$dropout.hidden[1] <- hyper_grid$dnn_dropout.hidden_1[j]
        if(!is.null(hyper_grid$dnn_dropout.hidden_2) && hyper_grid$dnn_dropout.hidden_2[j] > 0) layer$method_options$dropout.hidden[2] <- hyper_grid$dnn_dropout.hidden_2[j]
        if(!is.null(hyper_grid$dnn_dropout.hidden_3) && hyper_grid$dnn_dropout.hidden_3[j] > 0) layer$method_options$dropout.hidden[3] <- hyper_grid$dnn_dropout.hidden_3[j]

        if(!is.null(hyper_grid$batch_size) && hyper_grid$batch_size[j] > 0)
          layer$method_options$batch_size <- hyper_grid$batch_size[j]

      }

    }

    cat(sprintf("_ Overall best score is %s for hyper_grid row with ", best_score))
    cat(sprintf("%s=%s",names(hyper_grid),hyper_grid[best_j,]))
    cat('\n')

    layer$best_score <- best_score
    layer$hyper_grid <- hyper_grid
    layer$hyper_grid$mean_scores <- mean_scores
    layer$hyper_grid <- layer$hyper_grid %>% arrange(mean_scores)

  }
  else if(layer$method_options$bayesOpt) {

    Folds <- list()
    names <- c()
    folds <- caret::createFolds(y = y, k = layer$method_options$nfolds, list = F)
    data_baked$folds <- folds

    for(i in 1:layer$method_options$nfolds) {
      #Folds[[i]] <- as.integer(seq(i,nrows,by = layer$method_options$nfolds))
      Folds[[i]] <- which(data_baked$folds == i)
      names[i]   <- paste0('Fold',i)
    }
    names(Folds) <- names

    layer$init.weights <- T

    scoringFunction <- function(dnn_hidden_1, dnn_hidden_2, dnn_hidden_3,
                                dnn_dropout.hidden_1, dnn_dropout.hidden_2, dnn_dropout.hidden_3) {

      score <- c()

      for(k in 1:layer$method_options$nfolds) {

        if(!layer$method_options$use_embedding) {
          x.val         <- layer$x[Folds[[k]],] %>% as.matrix()
          x.glm.val     <- layer$x.glm[Folds[[k]],] %>% as.matrix()
        }
        else {

          x_fact.val <- NULL
          if(length(layer$fact_var)>0) {
            x_fact.val <- list()
            for(i in 1:length(layer$fact_var)) {
              x_fact.val[[i]] <- layer$x_fact[[i]][Folds[[k]]] %>% as.integer()
            }
          }

          x_no_fact.val <- NULL
          if(length(layer$no_fact_var)>0)
            x_no_fact.val  <- layer$x_no_fact[Folds[[k]],] %>% as.matrix()

          x_fact.glm.val <- NULL
          if(length(layer$fact_var.glm)>0) {
            x_fact.glm.val <- list()
            for(i in 1:length(layer$fact_var.glm)) {
              x_fact.glm.val[[i]] <- layer$x_fact.glm[[i]][Folds[[k]]] %>% as.integer()
            }
          }

          x_no_fact.glm.val <- NULL
          if(length(layer$no_fact_var.glm)>0)
            x_no_fact.glm.val <- layer$x_no_fact.glm[Folds[[k]],] %>% as.matrix()

        }

        y.val        <- layer$y[Folds[[k]]]
        sample.w.val <- weights.vec.n[Folds[[k]]]

        if(layer$method_options$nfolds==1) {

          if(!layer$method_options$use_embedding) {
            x     <- x.val
            x.glm <- x.glm.val
          }
          else {
            x_fact        <- x_fact.val
            x_no_fact     <- x_no_fact.val
            x_fact.glm    <- x_fact.glm.val
            x_no_fact.glm <- x_no_fact.glm.val
          }

          y <- y.val
          sample.w <- sample.w.val

        }
        else {

          if(!layer$method_options$use_embedding) {
            x      <- layer$x[-Folds[[k]],]
            x.glm  <- layer$x.glm[-Folds[[k]],]
          }
          else {
            x_fact <- NULL
            if(length(layer$fact_var)>0) {
              x_fact <- list()
              for(i in 1:length(layer$fact_var)) {
                x_fact[[i]] <- layer$x_fact[[i]][-Folds[[k]]] %>% as.integer()
              }
            }
            x_no_fact <- NULL
            if(length(layer$no_fact_var)>0)
              x_no_fact <- layer$x_no_fact[-Folds[[k]],] %>% as.matrix()

            x_fact.glm <- NULL
            if(length(layer$fact_var.glm)>0) {
              x_fact.glm <- list()
              for(i in 1:length(layer$fact_var.glm)) {
                x_fact.glm[[i]] <- layer$x_fact.glm[[i]][-Folds[[k]]] %>% as.integer()
              }
            }
            x_no_fact.glm <- NULL
            if(length(layer$no_fact_var.glm)>0)
              x_no_fact.glm <- layer$x_no_fact.glm[-Folds[[k]],] %>% as.matrix()
          }

          y <- layer$y[-Folds[[k]]]
          sample.w <- weights.vec.n[-Folds[[k]]]

        }

        def_inputs <- def_inputs(use_embedding=layer$method_options$use_embedding,
                                 x=x,
                                 model.glm=model.glm,
                                 no_fact_var=layer$no_fact_var,
                                 no_fact_var.glm=layer$no_fact_var.glm,
                                 fact_var.glm=layer$fact_var.glm,
                                 fact_var=layer$fact_var,
                                 x_fact=x_fact,
                                 x_fact.glm=x_fact.glm)

        # GLM Neural network

        if(!layer$method_options$use_embedding) {
          GLMNetwork.tmp <- def_inputs$inputs.glm %>%
            layer_dense(units=1, activation='linear', name='output_layer_glm', trainable=FALSE,
                        weights=list(array(model.glm$coefficients[2:length(model.glm$coefficients)], dim=c(length(model.glm$coefficients)-1,1)),
                                     array(model.glm$coefficients[1],dim=c(1))))
        }
        else {

          coef_no_fact.glm <- NULL
          if(length(def_inputs$beta.no_fact_var.glm) > 1)
            coef_no_fact.glm <- def_inputs$beta.no_fact_var.glm[2:length(def_inputs$beta.no_fact_var.glm)]

          coef_fact_var.glm <- NULL
          if(length(layer$fact_var.glm)>0)
            coef_fact_var.glm <- rep(1,length(layer$fact_var.glm))

          GLMNetwork.tmp <- def_inputs$inputs.glm %>%
            layer_dense(units=1, activation='linear', name='output_layer_glm', trainable=FALSE,
                        weights=list(array(c(coef_no_fact.glm,coef_fact_var.glm),
                                           dim=c(length(layer$fact_var.glm)+length(def_inputs$beta.no_fact_var.glm)-1,1)),
                                     array(def_inputs$beta.no_fact_var.glm[1],dim=c(1))))
        }

        # Hyperparameters

        dnn_hidden <- c()

        if(!is.null(layer$method_options$hidden))
          dnn_hidden <- layer$method_options$hidden

        if(!missing(dnn_hidden_1)) dnn_hidden[1] <- dnn_hidden_1 * layer$method_options$bayesOpt_step
        if(!missing(dnn_hidden_2)) dnn_hidden[2] <- dnn_hidden_2 * layer$method_options$bayesOpt_step
        if(!missing(dnn_hidden_3)) dnn_hidden[3] <- dnn_hidden_3 * layer$method_options$bayesOpt_step

        if(length(dnn_hidden)==0) dnn_hidden <- NULL

        dnn_dropout.hidden <- c()

        if(!is.null(layer$method_options$dropout.hidden))
          dnn_dropout.hidden <- layer$method_options$dropout.hidden
        else {
          if(!is.null(dnn_hidden))
            dnn_dropout.hidden <- rep(0,length(dnn_hidden))
        }

        if(!missing(dnn_dropout.hidden_1)) dnn_dropout.hidden[1] <- dnn_dropout.hidden_1
        if(!missing(dnn_dropout.hidden_2)) dnn_dropout.hidden[2] <- dnn_dropout.hidden_2
        if(!missing(dnn_dropout.hidden_3)) dnn_dropout.hidden[3] <- dnn_dropout.hidden_3

        if(length(dnn_dropout.hidden)==0) dnn_dropout.hidden <- NULL

        # Neural network

        NNetwork.tmp <- def_NN_arch(def_inputs$inputs,
                                    layer$method_options$batch_normalization,
                                    dnn_hidden,
                                    layer$method_options$activation.hidden,
                                    dnn_dropout.hidden,
                                    layer$method_options$activation.output,
                                    layer$method_options$use_bias)

        # CANN

        CANNoutput.tmp <- list(GLMNetwork.tmp, NNetwork.tmp) %>% layer_add() %>%
          layer_dense(units = 1, activation = layer$method_options$activation.output.cann, trainable = !layer$method_options$fixed.cann,
                      weights = switch(layer$method_options$fixed.cann + 1,NULL,list(array(c(1), dim=c(1,1)),
                                                                                     array(0, dim=c(1)))),
                      name = 'output_layer_CANN')

        if(!layer$method_options$use_embedding)
          CANN.tmp <- keras_model(inputs = c(def_inputs$inputs,
                                             def_inputs$inputs.glm),
                                  outputs = c(CANNoutput.tmp), name = 'CANN.tmp')
        else
          CANN.tmp <- keras_model(inputs = c(def_inputs$inputs_no_fact.glm,
                                             def_inputs$input_layer_emb.glm,
                                             def_inputs$inputs_no_fact,
                                             def_inputs$input_layer_emb),
                                  outputs = c(CANNoutput.tmp), name = 'CANN.tmp')

        CANN.tmp %>% compile(
          loss = layer$method_options$loss,
          optimizer = layer$method_options$optimizer,
          metrics = layer$method_options$metrics
        )

        if(k==1) weights.init <- keras::get_weights(CANN.tmp)
        else keras::set_weights(CANN.tmp, weights.init)

        earlystopping <- callback_early_stopping(
          monitor = layer$method_options$monitor,
          patience = layer$method_options$patience,
          restore_best_weights = T)

        if(!layer$method_options$use_embedding) {
          x.inputs <- list(x,x.glm)
          x.inputs.val <- list(x.val,x.glm.val)
        }
        else {
          x.inputs <- list(x_no_fact.glm,x_fact.glm,x_no_fact,x_fact)
          x.inputs[sapply(x.inputs, is.null)] <- NULL
          x.inputs.val <- list(x_no_fact.glm.val,x_fact.glm.val,x_no_fact.val,x_fact.val)
          x.inputs.val[sapply(x.inputs.val, is.null)] <- NULL
        }

        if(layer$method_options$nfolds > 1) {
          history.tmp <- CANN.tmp %>%
            keras::fit(x=x.inputs, y=y, sample_weight = sample.w, epochs = layer$method_options$epochs,
                       batch_size = layer$method_options$batch_size,
                       #validation_split = layer$method_options$validation_split,
                       validation_data = list(x.inputs.val,y.val,sample.w.val),
                       callbacks = list(earlystopping),
                       shuffle = layer$method_options$shuffle,
                       verbose = layer$method_options$verbose)
        }
        else {
          history.tmp <- CANN.tmp %>%
            keras::fit(x=x.inputs, y=y, sample_weight = sample.w, epochs = layer$method_options$epochs,
                       batch_size = layer$method_options$batch_size,
                       validation_split = layer$method_options$validation_split,
                       #validation_data = list(x.inputs.val,y.val,sample.w.val),
                       callbacks = list(earlystopping),
                       shuffle = layer$method_options$shuffle,
                       verbose = layer$method_options$verbose)
        }

        score[k] <- ifelse(layer$method_options$bayesOpt.min, -min(history.tmp$metrics[[4]]), max(history.tmp$metrics[[4]]))

      }

      return(list(Score = mean(score),
                  Pred = 0))

    }


    bounds <- layer$method_options$bayesOpt_bounds

    bounds_names <- as.vector(names(bounds))

    tNoPar <- system.time(
      optObj <- bayesOpt(
        FUN = scoringFunction
        , bounds = bounds
        #, initPoints = max(length(bounds)+1,3)
        , initPoints = layer$method_options$bayesOpt_initPoints
        , iters.n = layer$method_options$bayesOpt_iters_n
        , iters.k = 1
      )
    )

    # cl <- makeCluster(2)
    # registerDoParallel(cl)
    # clusterExport(cl=cl,varlist=c('Folds','data.xgb'),envir=environment())
    # clusterEvalQ(cl,expr= {
    #   library(xgboost)
    # })
    #
    # tWithPar <- system.time(
    #   optObj <- bayesOpt(
    #     FUN = scoringFunction
    #     , bounds = bounds
    #     , initPoints = 10
    #     , iters.n = layer$method_options$bayesOpt_iters_n
    #     , iters.k = 2
    #     , parallel = TRUE
    #   )
    # )
    # stopCluster(cl)
    # registerDoSEQ()

    layer$optObj <- optObj

    if(is.null(layer$method_options$hidden) &
       (('dnn_hidden_1' %in% bounds_names) | ('dnn_hidden_2' %in% bounds_names) | ('dnn_hidden_3' %in% bounds_names)) ) {
      layer$method_options$hidden <- c()
      #layer$method_options$bias_regularization <- TRUE
    }

    if('dnn_hidden_1' %in% bounds_names) layer$method_options$hidden[1] <- getBestPars(optObj)$dnn_hidden_1 * layer$method_options$bayesOpt_step
    if('dnn_hidden_2' %in% bounds_names) layer$method_options$hidden[2] <- getBestPars(optObj)$dnn_hidden_2 * layer$method_options$bayesOpt_step
    if('dnn_hidden_3' %in% bounds_names) layer$method_options$hidden[3] <- getBestPars(optObj)$dnn_hidden_3 * layer$method_options$bayesOpt_step

    if(is.null(layer$method_options$dropout.hidden) &
       (('dnn_dropout.hidden_1' %in% bounds_names) | ('dnn_dropout.hidden_2' %in% bounds_names) | ('dnn_dropout.hidden_3' %in% bounds_names)) )
      layer$method_options$dropout.hidden <- c()

    if('dnn_dropout.hidden_1' %in% bounds_names) layer$method_options$dropout.hidden[1] <- getBestPars(optObj)$dnn_dropout.hidden_1
    if('dnn_dropout.hidden_2' %in% bounds_names) layer$method_options$dropout.hidden[2] <- getBestPars(optObj)$dnn_dropout.hidden_2
    if('dnn_dropout.hidden_3' %in% bounds_names) layer$method_options$dropout.hidden[3] <- getBestPars(optObj)$dnn_dropout.hidden_3

  }

  x        <- layer$x
  x.glm    <- layer$x.glm

  if(layer$method_options$use_embedding) {
    x_fact        <-layer$x_fact
    x_no_fact     <-layer$x_no_fact
    x_fact.glm    <-layer$x_fact.glm
    x_no_fact.glm <-layer$x_no_fact.glm
    no_fact_var   <-layer$no_fact_var
    no_fact_var.glm <- layer$no_fact_var.glm
    fact_var.glm  <- layer$fact_var.glm
    fact_var      <- layer$fact_var
  }

  y        <- layer$y
  sample.w <- weights.vec.n


  def_inputs <- def_inputs(use_embedding=layer$method_options$use_embedding,
                           x=x,
                           model.glm=model.glm,
                           no_fact_var=no_fact_var,
                           no_fact_var.glm=no_fact_var.glm,
                           fact_var.glm=fact_var.glm,
                           fact_var=fact_var,
                           x_fact=x_fact,
                           x_fact.glm=x_fact.glm)

  # GLM Neural network

  if(!layer$method_options$use_embedding) {
    GLMNetwork <- def_inputs$inputs.glm %>%
      layer_dense(units=1, activation='linear', name='output_layer_glm', trainable=FALSE,
                  weights=list(array(model.glm$coefficients[2:length(model.glm$coefficients)], dim=c(length(model.glm$coefficients)-1,1)),
                               array(model.glm$coefficients[1],dim=c(1))))
  }
  else {
    coef_no_fact.glm <- NULL
    if(length(def_inputs$beta.no_fact_var.glm) > 1)
      coef_no_fact.glm <- def_inputs$beta.no_fact_var.glm[2:length(def_inputs$beta.no_fact_var.glm)]

    coef_fact_var.glm <- NULL
    if(length(fact_var.glm)>0)
      coef_fact_var.glm <- rep(1,length(fact_var.glm))

    GLMNetwork <- def_inputs$inputs.glm %>%
      layer_dense(units=1, activation='linear', name='output_layer_glm', trainable=FALSE,
                  weights=list(array(c(coef_no_fact.glm,coef_fact_var.glm),
                                     dim=c(length(fact_var.glm)+length(def_inputs$beta.no_fact_var.glm)-1,1)),
                               array(def_inputs$beta.no_fact_var.glm[1],dim=c(1))))
  }

  # Neural network

  NNetwork <- def_NN_arch(inputs=def_inputs$inputs,
                          batch_normalization=layer$method_options$batch_normalization,
                          hidden=layer$method_options$hidden,
                          activation.hidden=layer$method_options$activation.hidden,
                          dropout.hidden=layer$method_options$dropout.hidden,
                          activation.output=layer$method_options$activation.output,
                          use_bias=layer$method_options$use_bias)

  # CANN

  CANNoutput <- list(GLMNetwork, NNetwork) %>% layer_add() %>%
    layer_dense(units = 1, activation = layer$method_options$activation.output.cann, trainable = !layer$method_options$fixed.cann,
                weights = switch(layer$method_options$fixed.cann + 1,NULL,list(array(c(1), dim=c(1,1)),
                                                                               array(0, dim=c(1)))),
                name = 'output_layer_CANN')

  if(!layer$method_options$use_embedding)
    CANN <- keras_model(inputs = c(def_inputs$inputs,
                                   def_inputs$inputs.glm),
                        outputs = c(CANNoutput), name = 'CANN')
  else
    CANN <- keras_model(inputs = c(def_inputs$inputs_no_fact.glm,
                                   def_inputs$input_layer_emb.glm,
                                   def_inputs$inputs_no_fact,
                                   def_inputs$input_layer_emb),
                        outputs = c(CANNoutput), name = 'CANN')

  print(summary(CANN))

  CANN %>% compile(
    loss = layer$method_options$loss,
    optimizer = layer$method_options$optimizer,
    metrics = layer$method_options$metrics
  )

  if(layer$method_options$gridsearch_cv) keras::set_weights(CANN, init_weights[[best_j]])

  earlystopping <- callback_early_stopping(
    monitor = layer$method_options$monitor,
    patience = layer$method_options$patience,
    restore_best_weights = T)

  indVal <- caret::createDataPartition(y=y,p=layer$method_options$validation_split, list=FALSE)

  if(!layer$method_options$use_embedding) {
    x.val         <- layer$x[indVal,] %>% as.matrix()
    x.glm.val     <- layer$x.glm[indVal,] %>% as.matrix()
  }
  else {

    x_fact.val <- NULL
    if(length(layer$fact_var)>0) {
      x_fact.val <- list()
      for(i in 1:length(layer$fact_var)) {
        x_fact.val[[i]] <- layer$x_fact[[i]][indVal] %>% as.integer()
      }
    }

    x_no_fact.val <- NULL
    if(length(layer$no_fact_var)>0)
      x_no_fact.val  <- layer$x_no_fact[indVal,] %>% as.matrix()

    x_fact.glm.val <- NULL
    if(length(layer$fact_var.glm)>0) {
      x_fact.glm.val <- list()
      for(i in 1:length(layer$fact_var.glm)) {
        x_fact.glm.val[[i]] <- layer$x_fact.glm[[i]][indVal] %>% as.integer()
      }
    }

    x_no_fact.glm.val <- NULL
    if(length(layer$no_fact_var.glm)>0)
      x_no_fact.glm.val <- layer$x_no_fact.glm[indVal,] %>% as.matrix()

  }

  y.val        <- layer$y[indVal]
  sample.w.val <- weights.vec.n[indVal]

  if(!layer$method_options$use_embedding) {
    x      <- layer$x[-indVal,]
    x.glm  <- layer$x.glm[-indVal,]
  }
  else {
    x_fact <- NULL
    if(length(layer$fact_var)>0) {
      x_fact <- list()
      for(i in 1:length(layer$fact_var)) {
        x_fact[[i]] <- layer$x_fact[[i]][-indVal] %>% as.integer()
      }
    }
    x_no_fact <- NULL
    if(length(layer$no_fact_var)>0)
      x_no_fact <- layer$x_no_fact[-indVal,] %>% as.matrix()

    x_fact.glm <- NULL
    if(length(layer$fact_var.glm)>0) {
      x_fact.glm <- list()
      for(i in 1:length(layer$fact_var.glm)) {
        x_fact.glm[[i]] <- layer$x_fact.glm[[i]][-indVal] %>% as.integer()
      }
    }
    x_no_fact.glm <- NULL
    if(length(layer$no_fact_var.glm)>0)
      x_no_fact.glm <- layer$x_no_fact.glm[-indVal,] %>% as.matrix()
  }

  y <- layer$y[-indVal]
  sample.w <- weights.vec.n[-indVal]

  if(!layer$method_options$use_embedding) {
    x.inputs <- list(x,x.glm)
    x.inputs.val <- list(x.val,x.glm.val)
  }
  else {
    x.inputs <- list(x_no_fact.glm,x_fact.glm,x_no_fact,x_fact)
    x.inputs[sapply(x.inputs, is.null)] <- NULL
    x.inputs.val <- list(x_no_fact.glm.val,x_fact.glm.val,x_no_fact.val,x_fact.val)
    x.inputs.val[sapply(x.inputs.val, is.null)] <- NULL
  }

  # if(!layer$method_options$use_embedding) {
  #   x.inputs <- list(x,x.glm)
  # }
  # else {
  #   x.inputs <- list(x_no_fact.glm,x_fact.glm,x_no_fact,x_fact)
  #   x.inputs[sapply(x.inputs, is.null)] <- NULL
  # }

  #now <- Sys.time()
  #fn <- paste0("./tmp/cann_best_weights_",format(now, "%Y%m%d_%H%M%S.hdf5"))

  #CBs <- callback_model_checkpoint(fn, monitor=layer$method_options$monitor, save_best_only = TRUE, save_weights_only = TRUE)

  layer$history <- CANN %>%
    keras::fit(x=x.inputs, y=y, sample_weight=sample.w, epochs = layer$method_options$epochs,
               batch_size = layer$method_options$batch_size,
               validation_data = list(x.inputs.val,y.val,sample.w.val),
               #validation_split = layer$method_options$validation_split,
               #callbacks = list(earlystopping,CBs),
               callbacks = list(earlystopping),
               shuffle = layer$method_options$shuffle,
               verbose = layer$method_options$verbose)

  #load_model_weights_hdf5(CANN, fn)

  if(!layer$method_options$bias_regularization) {
  layer$fit <- CANN
  }
  else {
    # We keep track of the neural network (biased) model
    layer$fit.biased <- CANN
    # Source: Ferrario, Andrea and Noll, Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April 23, 2020), p.52
    glm.formula <- function(nb) {
      string <- "yy ~ glm.pred + X1"
      #string <- "yy ~ X1"
      if(nb>1) {for (i in 2:nb) {string <- paste(string," + X",i, sep="")}}
      string
    }

    glm.formula.2 <- function(cov) {
      string <- "yy ~ "
      for (i in 2:length(cov)) {
        if(i==2) string <- paste0(string,' ',cov[i])
        else string <- paste0(string,' + ',cov[i])
      }
      string
    }

    #zz        <- keras_model(inputs = CANN$input, outputs=get_layer(CANN,'output_layer_CANN')$output)
    zz        <- keras_model(inputs = CANN$input, outputs=get_layer(CANN,'output_layer_NN')$output)
    #zz        <- keras_model(inputs = CANN$input, outputs=get_layer(CANN,'last_hidden_layer_activation')$output)
    layer$zz  <- zz

    #glm.pred <- layer$model.glm$fitted.values
    glm.pred <- layer$model.glm$linear.predictors

    if(!layer$method_options$use_embedding) {
      x     <- layer$x
      x.glm <- layer$x.glm
    }
    else {
      x_fact        <-layer$x_fact
      x_no_fact     <-layer$x_no_fact
      x_fact.glm    <-layer$x_fact.glm
      x_no_fact.glm <-layer$x_no_fact.glm
      no_fact_var   <-layer$no_fact_var
      no_fact_var.glm <- layer$no_fact_var.glm
      fact_var.glm  <- layer$fact_var.glm
      fact_var      <- layer$fact_var
    }

    y <- layer$y

    if(!layer$method_options$use_embedding) {
      x.inputs <- list(x,x.glm)
    }
    else {
      x.inputs <- list(x_no_fact.glm,x_fact.glm,x_no_fact,x_fact)
      x.inputs[sapply(x.inputs, is.null)] <- NULL
    }

    Zlearn    <- data.frame(zz %>% predict(x.inputs))
    names(Zlearn) <- paste0('X', 1:ncol(Zlearn))

    Zlearn$yy <- y
    Zlearn$glm.pred <- glm.pred
    data$glm.pred <- glm.pred

    # We keep track of the pre-processed data for analysis purposes
    layer$Zlearn <- Zlearn

    if(layer$method_options$distribution == 'gamma')
      fam <- Gamma(link=log) # default link=inverse but we use exponential as activation function
    else if(layer$method_options$distribution == 'bernoulli')
      fam <- binomial() # default link = logit <-> activation function = sigmoid
    else if(layer$method_options$distribution == 'poisson')
      fam <- poisson() # default link = log <-> activation function = exponential
    else if(layer$method_options$distribution == 'gaussian')
      fam <- gaussian() # default link = identity <-> activation function = identity
    else
      stop('Bias regularization is not supported for this distribution.')

    glm1 <- glm(as.formula(glm.formula(ncol(Zlearn)-2)), data=Zlearn, family=fam, weights = weights.vec.n)
    #glm1 <- glm(as.formula(glm.formula(ncol(Zlearn)-1)), data=Zlearn, family=fam, weights = weights.vec.n)
    cov <- names(glm1$coefficients[!sapply(glm1$coefficients,is.na)])
    if(length(cov)>0) {
      glm2 <- glm(as.formula(glm.formula.2(cov)), data=Zlearn, family=fam, weights = weights.vec.n)
      layer$fit <- glm2
    }
    else layer$fit <- glm1
  }

  if(!is.null(obj$balance.var)){
    layer$balance.correction <- sapply(data %>% group_split(data[[obj$balance.var]]),
                                       function(x) {

                                         data_baked_bc     <- bake(layer$data_recipe, new_data = x)
                                         if(ncol(data_baked_bc) == 1)
                                           data_baked_bc <- data_baked_bc %>% mutate(intercept = 1)

                                         data_baked_bc.glm <- bake(layer$data_recipe.glm, new_data = x)

                                         if(ncol(data_baked_bc.glm) == 1)
                                           data_baked_bc.glm <- data_baked_bc.glm %>% mutate(intercept = 1)

                                         def_x <- def_x(layer$method_options$use_embedding,
                                                        layer$method_options$embedding_var,
                                                        layer$method_options$embedding_var.glm,
                                                        f,
                                                        f.glm,
                                                        x,
                                                        data_baked_bc,
                                                        data_baked_bc.glm,
                                                        layer$data_recipe,
                                                        layer$data_recipe.glm,
                                                        label)

                                         if(!layer$method_options$use_embedding) {
                                           x.tmp <- def_x$x
                                           x.tmp.glm <- def_x$x.glm
                                           x.inputs.tmp <- list(x.tmp,x.tmp.glm)
                                         }
                                         else {
                                           x_fact.tmp         <- def_x$x_fact
                                           x_no_fact.tmp      <- def_x$x_no_fact
                                           x_fact.glm.tmp     <- def_x$x_fact.glm
                                           x_no_fact.glm.tmp  <- def_x$x_no_fact.glm
                                           x.inputs.tmp <- list(x_no_fact.glm.tmp,x_fact.glm.tmp,x_no_fact.tmp,x_fact.tmp)
                                           x.inputs.tmp[sapply(x.inputs.tmp, is.null)] <- NULL
                                         }

                                         if(layer$method_options$bias_regularization) {
                                           Zlearn.tmp   <- data.frame(layer$zz %>% predict(x.inputs.tmp))
                                           names(Zlearn.tmp) <- paste0('X', 1:ncol(Zlearn.tmp))
                                           Zlearn.tmp$glm.pred <- x$glm.pred
                                           sum(x[[layer$name]])/sum(predict(layer$fit, newdata = Zlearn.tmp, type = 'response'))
                                         }
                                         else {
                                           sum(x[[layer$name]])/sum(layer$fit %>% predict(x.inputs.tmp))
                                         }

                                       })
  }

  if(layer$method_options$bias_regularization)
    pred <- layer$fit$fitted.values
  else
    pred <- layer$fit %>% predict(x.inputs)

  y <- layer$y

  if(layer$method_options$distribution == 'gaussian') {
    layer$sigma <- sd(pred - y)
  }

  if(layer$method_options$distribution == 'gamma') {

    if(is.null(obj$weights)) weights.vec.normalized <- NULL
    else weights.vec.normalized <- weights.vec*length(weights.vec)/sum(weights.vec)

    shape <- hirem_gamma_shape(observed = y, fitted = pred, weight = weights.vec.normalized)
    layer$shape <- shape$shape
    layer$shape.se <- shape$se
    layer$weights.vec.normalized <- weights.vec.normalized
  }

  return(layer)
}
