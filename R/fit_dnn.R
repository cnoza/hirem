

#' @import tensorflow
#' @import keras
#' @importFrom tidyselect all_of
#' @importFrom data.table fsetdiff as.data.table
#' @importFrom recipes recipe step_log step_normalize step_dummy bake prep all_nominal all_numeric all_outcomes
#' @export
fit.layer_dnn <- function(layer, obj, formula, training = FALSE, fold = NULL) {

  cat(sprintf("Fitting layer_dnn for %s...\n", layer$name))

  layer$formula <- formula

  data <- obj$data_training
  if(!training) {
    data <- obj$data_observed
  }

  if(!is.null(fold)) {
    data <- data %>% filter(cv_fold != fold)
  }

  data <- data[layer$filter(data), ]

  f <- as.formula(formula)
  label <- as.character(terms(f)[[2]])

  if(!is.null(layer$transformation)) {
    data[,label] <- layer$transformation$transform(data[,label])
  }

  weights.vec <- if(is.null(obj$weights) | !is.null(layer$method_options$ae.hidden)) NULL else obj$weights[data[[obj$weight.var]]]

  if(!is.null(weights.vec)) weights.vec.n <- weights.vec*length(weights.vec)/sum(weights.vec)
  else weights.vec.n <- NULL

  # for testing purposes...
  weights.vec.n <- NULL

  layer$weights.vec.n <- weights.vec.n

  data_recipe <- recipe(f, data=data)

  if(layer$method_options$step_log)
    data_recipe <- data_recipe %>% step_log(as.name(label))
  if(layer$method_options$step_normalize)
    data_recipe <- data_recipe %>% step_normalize(all_numeric(), -all_outcomes())
  if(!layer$method_options$use_embedding)
    data_recipe <- data_recipe %>% step_dummy(all_nominal(), one_hot = layer$method_options$one_hot)

  data_recipe <- data_recipe %>% prep()
  layer$data_recipe <- data_recipe

  data_baked <- bake(data_recipe, new_data = data)
  if(ncol(data_baked) == 1)
    data_baked <- data_baked %>% mutate(intercept = 1)

  x_fact <- NULL
  x_no_fact <- NULL

  def_x <- def_x_mlp(layer$method_options$use_embedding,
                     layer$method_options$embedding_var,
                     f,
                     data,
                     data_baked,
                     data_recipe,
                     label)

  x       <- def_x$x
  layer$x <- x

  if(layer$method_options$use_embedding) {
    layer$x_fact         <- def_x$x_fact
    layer$x_no_fact      <- def_x$x_no_fact
    layer$fact_var       <- def_x$fact_var
    layer$no_fact_var    <- def_x$no_fact_var
  }

  y <- data_baked %>% pull(as.name(label))

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

    hyper_grid <- layer$method_options$hyper_grid
    if(is.null(hyper_grid)) {
      hyper_grid <- expand.grid(
        dnn_hidden_1 = seq(from = 0, to = 30, by = 10)
      , dnn_hidden_2 = seq(from = 0, to = 30, by = 10)
      , dnn_hidden_3 = seq(from = 0, to = 30, by = 10)
      )
    }

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

    if(layer$method_options$random_trials > 0) {
      i <- sample(1:dim(hyper_grid)[1],layer$method_options$random_trials)
      hyper_grid <- hyper_grid[i,]
    }

    cat('_ Initializing the weights...\n')

    init_weights <- list()

    for(j in seq_len(nrow(hyper_grid))) {

      if(!layer$method_options$use_embedding) {

        ae.hidden <- c()

        if(!is.null(layer$method_options$ae.hidden))
          ae.hidden <- layer$method_options$ae.hidden

        if(!is.null(hyper_grid$ae_hidden_1)) ae.hidden[1] <- hyper_grid$ae_hidden_1[j]
        if(!is.null(hyper_grid$ae_hidden_2)) ae.hidden[2] <- hyper_grid$ae_hidden_2[j]
        if(!is.null(hyper_grid$ae_hidden_3)) ae.hidden[3] <- hyper_grid$ae_hidden_3[j]

        if(length(ae.hidden)==0) ae.hidden <- NULL

        inputs <- layer_input(shape = c(ncol(x)), name='input_layer')

        if(!is.null(ae.hidden)) {

          ae_arch <- def_ae_arch(inputs=inputs,
                                 x=x,
                                 ae.hidden=ae.hidden,
                                 ae.activation.hidden=layer$method_options$ae.activation.hidden)

          autoencoder <- keras_model(inputs, ae_arch$ae_output_l)

          model_en <- keras_model(inputs, ae_arch$ae_hidden_l[[length(ae.hidden)]])

          autoencoder %>% compile(loss = 'mae', optimizer='adam')

          autoencoder %>% keras::fit(
            x = x,
            y = x,
            epochs = layer$method_options$epochs,
            batch_size = layer$method_options$batch_size,
            verbose = layer$method_options$verbose,
            shuffle = layer$method_options$shuffle
          )

          x <- model_en %>% predict(x)

          inputs <- layer_input(shape = c(ncol(x)), name='ae_input_layer')

          x.sav <- x

        }

      }

      #for(k in 1:layer$method_options$nfolds) {

      # We only need to do this for 1 fold
      k <- 1

      if(!layer$method_options$use_embedding) {
        if(!is.null(ae.hidden))
          x.val    <- x.sav[Folds[[k]],]
        else
          x.val    <- layer$x[Folds[[k]],]
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
      }

      y.val    <- layer$y[Folds[[k]]]
      sample.w.val <- weights.vec.n[Folds[[k]]]

      if(layer$method_options$nfolds==1) {
        if(!layer$method_options$use_embedding) {
          x    <- x.val
        }
        else {
          x_fact        <- x_fact.val
          x_no_fact     <- x_no_fact.val
        }
        y    <- y.val
        sample.w <- sample.w.val
      }
      else {

        if(!layer$method_options$use_embedding) {
          if(!is.null(ae.hidden))
            x <- x.sav[-Folds[[k]],]
          else
            x <- layer$x[-Folds[[k]],]
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
        }

        y <- layer$y[-Folds[[k]]]
        sample.w <- weights.vec.n[-Folds[[k]]]

      }

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

      batch_size <- layer$method_options$batch_size
      if(!is.null(hyper_grid$batch_size)) batch_size <- hyper_grid$batch_size[j]

      lr <- NULL
      if(!is.null(hyper_grid$learning_rate) && hyper_grid$learning_rate[j] > 0) lr <- hyper_grid$learning_rate[j]

      def_inputs <- def_inputs_mlp(use_embedding=layer$method_options$use_embedding,
                                   x=x,
                                   no_fact_var=layer$no_fact_var,
                                   fact_var=layer$fact_var,
                                   x_fact=x_fact,
                                   output_dim = layer$method_options$output_dim)

      dnn_arch <- def_dnn_arch(inputs=def_inputs$inputs,
                               batch_normalization=layer$method_options$batch_normalization,
                               hidden=dnn_hidden,
                               activation.hidden=layer$method_options$activation.hidden,
                               dropout.hidden=dnn_dropout.hidden,
                               family_for_init=layer$method_options$family_for_init,
                               label=label,
                               data=data,
                               activation.output=layer$method_options$activation.output,
                               x=x,
                               use_bias=layer$method_options$use_bias,
                               weights.vec=weights.vec.n,
                               x_fact = x_fact,
                               x_no_fact = x_no_fact,
                               output_dim = layer$method_options$output_dim)

      if(!layer$method_options$use_embedding) {
        model <- keras_model(inputs = def_inputs$inputs, outputs = c(dnn_arch$output))
      }
      else {
        model <- keras_model(inputs = c(def_inputs$inputs_no_fact,
                                        def_inputs$input_layer_emb),
                             outputs = c(dnn_arch$output))
      }

      if(!is.null(lr)) {
        if(layer$method_options$optimizer == 'nadam') {
          optim <- optimizer_nadam(lr = lr)
        }
        else if(layer$method_options$optimizer == 'adam') {
          optim <- optimizer_adam(lr = lr)
        }
        else {
          optim <- layer$method_options$optimizer
        }
      }
      else {
        optim <- layer$method_options$optimizer
      }

      model %>% compile(
        loss = layer$method_options$loss,
        optimizer = optim,
        metrics = layer$method_options$metrics
      )

      if(layer$method_options$verbose == 1) print(summary(model))

      init_weights[[j]] <- keras::get_weights(model)

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

      if(!layer$method_options$use_embedding) {

        ae.hidden <- c()

        if(!is.null(layer$method_options$ae.hidden))
          ae.hidden <- layer$method_options$ae.hidden

        if(!is.null(hyper_grid$ae_hidden_1)) ae.hidden[1] <- hyper_grid$ae_hidden_1[j]
        if(!is.null(hyper_grid$ae_hidden_2)) ae.hidden[2] <- hyper_grid$ae_hidden_2[j]
        if(!is.null(hyper_grid$ae_hidden_3)) ae.hidden[3] <- hyper_grid$ae_hidden_3[j]

        if(length(ae.hidden)==0) ae.hidden <- NULL

        inputs <- layer_input(shape = c(ncol(x)), name='input_layer')

        if(!is.null(ae.hidden)) {

          ae_arch <- def_ae_arch(inputs=inputs,
                                 x=x,
                                 ae.hidden=ae.hidden,
                                 ae.activation.hidden=layer$method_options$ae.activation.hidden)

          autoencoder <- keras_model(inputs, ae_arch$ae_output_l)

          model_en <- keras_model(inputs, ae_arch$ae_hidden_l[[length(ae.hidden)]])

          autoencoder %>% compile(loss = 'mae', optimizer='adam')

          autoencoder %>% keras::fit(
            x = x,
            y = x,
            epochs = layer$method_options$epochs,
            batch_size = layer$method_options$batch_size,
            verbose = layer$method_options$verbose,
            shuffle = layer$method_options$shuffle
          )

          x <- model_en %>% predict(x)

          inputs <- layer_input(shape = c(ncol(x)), name='ae_input_layer')

          x.sav <- x

        }

      }

      score <- c()

      for(k in 1:layer$method_options$nfolds) {

        if(!layer$method_options$use_embedding) {
          if(!is.null(ae.hidden))
            x.val    <- x.sav[Folds[[k]],]
          else
            x.val    <- layer$x[Folds[[k]],]
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
        }

        y.val    <- layer$y[Folds[[k]]]
        sample.w.val <- weights.vec.n[Folds[[k]]]

        if(layer$method_options$nfolds==1) {
          if(!layer$method_options$use_embedding) {
            x    <- x.val
          }
          else {
            x_fact        <- x_fact.val
            x_no_fact     <- x_no_fact.val
          }
          y    <- y.val
          sample.w <- sample.w.val
        }
        else {

          if(!layer$method_options$use_embedding) {
            if(!is.null(ae.hidden))
              x <- x.sav[-Folds[[k]],]
            else
              x <- layer$x[-Folds[[k]],]
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
          }

          y <- layer$y[-Folds[[k]]]
          sample.w <- weights.vec.n[-Folds[[k]]]

        }

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

        batch_size <- layer$method_options$batch_size
        if(!is.null(hyper_grid$batch_size)) batch_size <- hyper_grid$batch_size[j]

        def_inputs <- def_inputs_mlp(use_embedding=layer$method_options$use_embedding,
                                     x=x,
                                     no_fact_var=layer$no_fact_var,
                                     fact_var=layer$fact_var,
                                     x_fact=x_fact,
                                     output_dim = layer$method_options$output_dim)

        dnn_arch <- def_dnn_arch(inputs=def_inputs$inputs,
                                 batch_normalization=layer$method_options$batch_normalization,
                                 hidden=dnn_hidden,
                                 activation.hidden=layer$method_options$activation.hidden,
                                 dropout.hidden=dnn_dropout.hidden,
                                 family_for_init=layer$method_options$family_for_init,
                                 label=label,
                                 data=data,
                                 activation.output=layer$method_options$activation.output,
                                 x=x,
                                 use_bias=layer$method_options$use_bias,
                                 weights.vec=weights.vec.n,
                                 x_fact = x_fact,
                                 x_no_fact = x_no_fact,
                                 output_dim = layer$method_options$output_dim)

        if(!layer$method_options$use_embedding) {
          model <- keras_model(inputs = def_inputs$inputs, outputs = c(dnn_arch$output))
        }
        else {
          model <- keras_model(inputs = c(def_inputs$inputs_no_fact,
                                          def_inputs$input_layer_emb),
                               outputs = c(dnn_arch$output))
        }

        model %>% compile(
          loss = layer$method_options$loss,
          optimizer = layer$method_options$optimizer,
          metrics = layer$method_options$metrics
        )

        if(layer$method_options$verbose == 1) print(summary(model))

        keras::set_weights(model, init_weights[[j]])

        # earlystopping <- callback_early_stopping(
        #   monitor = layer$method_options$monitor,
        #   patience = layer$method_options$patience,
        #   restore_best_weights = T)


        if(!layer$method_options$use_embedding) {
          x.inputs <- list(x)
          x.inputs.val <- list(x.val)
        }
        else {
          x.inputs <- list(x_no_fact,x_fact)
          x.inputs[sapply(x.inputs, is.null)] <- NULL
          x.inputs.val <- list(x_no_fact.val,x_fact.val)
          x.inputs.val[sapply(x.inputs.val, is.null)] <- NULL
        }

        #print(layer$method_options$batch_size)

        if(layer$method_options$nfolds > 1) {
          history <- model %>%
            keras::fit(x=x.inputs, y=y, sample_weight = sample.w, epochs = layer$method_options$epochs,
                       batch_size = batch_size,
                       #validation_split = layer$method_options$validation_split,
                       validation_data = list(x.inputs.val,y.val,sample.w.val),
                       callbacks = list(earlystopping),
                       shuffle = layer$method_options$shuffle,
                       verbose = layer$method_options$verbose)
        }
        else {
          history <- model %>%
            keras::fit(x=x.inputs, y=y, sample_weight = sample.w, epochs = layer$method_options$epochs,
                       batch_size = batch_size,
                       validation_split = layer$method_options$validation_split,
                       #validation_data = list(x.inputs.val,y.val,sample.w.val),
                       callbacks = list(earlystopping),
                       shuffle = layer$method_options$shuffle,
                       verbose = layer$method_options$verbose)
        }

        score[k] <- ifelse(layer$method_options$gridsearch_cv.min, min(history$metrics[[4]]), max(history$metrics[[4]]))

        cat(sprintf('_ Score for fold %s is %s\n',k,score[k]))

      }

      mean_score <- mean(score)
      cat(sprintf('_ Mean score for hyper_grid row %s is %s\n',j,mean_score))

      mean_scores[j] <- mean_score


      if(ifelse(layer$method_options$gridsearch_cv.min, mean_score < best_score, mean_score > best_score)) {

          best_score <- mean_score
          best_j <- j

          cat(sprintf("_ Best score so far is %s for hyper_grid row with ", best_score))
          cat(sprintf("%s=%s",names(hyper_grid),hyper_grid[best_j,]))
          cat('\n')

          if(is.null(layer$method_options$ae.hidden) &
             (!is.null(hyper_grid$ae_hidden_1) | !is.null(hyper_grid$ae_hidden_2) | !is.null(hyper_grid$ae_hidden_3))) {
            layer$method_options$ae.hidden <- c()
          }

          if(!is.null(hyper_grid$ae_hidden_1)) layer$method_options$ae.hidden[1] <- hyper_grid$ae_hidden_1[j]
          if(!is.null(hyper_grid$ae_hidden_2)) layer$method_options$ae.hidden[2] <- hyper_grid$ae_hidden_2[j]
          if(!is.null(hyper_grid$ae_hidden_3)) layer$method_options$ae.hidden[3] <- hyper_grid$ae_hidden_3[j]

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
      #Folds[[i]] <- as.integer(seq(i,nrow(x),by = layer$method_options$nfolds))
      Folds[[i]] <- which(data_baked$folds == i)
      names[i] <- paste0('Fold',i)
    }
    names(Folds) <- names

    scoringFunction <- function(ae_hidden_1, ae_hidden_2, ae_hidden_3,
                                dnn_hidden_1, dnn_hidden_2, dnn_hidden_3,
                                dnn_dropout.hidden_1, dnn_dropout.hidden_2, dnn_dropout.hidden_3,
                                batch_size) {

      if(!layer$method_options$use_embedding) {

        ae.hidden <- c()

        if(!is.null(layer$method_options$ae.hidden))
          ae.hidden <- layer$method_options$ae.hidden

        if(!missing(ae_hidden_1)) ae.hidden[1] <- ae_hidden_1
        if(!missing(ae_hidden_2)) ae.hidden[2] <- ae_hidden_2
        if(!missing(ae_hidden_3)) ae.hidden[3] <- ae_hidden_3

        if(length(ae.hidden)==0) ae.hidden <- NULL

        inputs <- layer_input(shape = c(ncol(x)), name='input_layer')

        if(!is.null(ae.hidden)) {

          ae_arch <- def_ae_arch(inputs=inputs,
                                 x=x,
                                 ae.hidden=ae.hidden,
                                 ae.activation.hidden=layer$method_options$ae.activation.hidden)

          autoencoder <- keras_model(inputs, ae_arch$ae_output_l)

          model_en <- keras_model(inputs, ae_arch$ae_hidden_l[[length(ae.hidden)]])

          autoencoder %>% compile(loss = 'mae', optimizer='adam')

          autoencoder %>% keras::fit(
            x = x,
            y = x,
            epochs = layer$method_options$epochs,
            batch_size = layer$method_options$batch_size,
            verbose = layer$method_options$verbose,
            shuffle = layer$method_options$shuffle
          )

          x <- model_en %>% predict(x)

          inputs <- layer_input(shape = c(ncol(x)), name='ae_input_layer')

          x.sav <- x

        }

      }

      score <- c()

      for(k in 1:layer$method_options$nfolds) {

        if(!layer$method_options$use_embedding) {
          if(!is.null(ae.hidden))
            x.val    <- x.sav[Folds[[k]],]
          else
            x.val    <- layer$x[Folds[[k]],]
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
        }

        y.val    <- layer$y[Folds[[k]]]
        sample.w.val <- weights.vec.n[Folds[[k]]]

        if(layer$method_options$nfolds==1) {
          if(!layer$method_options$use_embedding) {
            x    <- x.val
          }
          else {
            x_fact        <- x_fact.val
            x_no_fact     <- x_no_fact.val
          }
          y    <- y.val
          sample.w <- sample.w.val
        }
        else {

          if(!layer$method_options$use_embedding) {
            if(!is.null(ae.hidden))
              x <- x.sav[-Folds[[k]],]
            else
              x <- layer$x[-Folds[[k]],]
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
          }

          y <- layer$y[-Folds[[k]]]
          sample.w <- weights.vec.n[-Folds[[k]]]

        }

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

        batch_size_tmp <- layer$method_options$batch_size
        if(!missing(batch_size)) batch_size_tmp <- batch_size*layer$method_options$batch_size

        def_inputs <- def_inputs_mlp(use_embedding=layer$method_options$use_embedding,
                                     x=x,
                                     no_fact_var=layer$no_fact_var,
                                     fact_var=layer$fact_var,
                                     x_fact=x_fact,
                                     output_dim = layer$method_options$output_dim)

        dnn_arch <- def_dnn_arch(inputs=def_inputs$inputs,
                                 batch_normalization=layer$method_options$batch_normalization,
                                 hidden=dnn_hidden,
                                 activation.hidden=layer$method_options$activation.hidden,
                                 dropout.hidden=dnn_dropout.hidden,
                                 family_for_init=layer$method_options$family_for_init,
                                 label=label,
                                 data=data,
                                 activation.output=layer$method_options$activation.output,
                                 x=x,
                                 use_bias=layer$method_options$use_bias,
                                 weights.vec=weights.vec.n,
                                 x_fact = x_fact,
                                 x_no_fact = x_no_fact,
                                 output_dim = layer$method_options$output_dim)

        if(!layer$method_options$use_embedding) {
          model <- keras_model(inputs = def_inputs$inputs, outputs = c(dnn_arch$output))
        }
        else {
          model <- keras_model(inputs = c(def_inputs$inputs_no_fact,
                                          def_inputs$input_layer_emb),
                               outputs = c(dnn_arch$output))
        }

        model %>% compile(
          loss = layer$method_options$loss,
          optimizer = layer$method_options$optimizer,
          metrics = layer$method_options$metrics
        )

        if(k==1) weights.init <- keras::get_weights(model)
        else keras::set_weights(model, weights.init)

        earlystopping <- callback_early_stopping(
          monitor = layer$method_options$monitor,
          patience = layer$method_options$patience,
          restore_best_weights = T)

        if(!layer$method_options$use_embedding) {
          x.inputs <- list(x)
          x.inputs.val <- list(x.val)
        }
        else {
          x.inputs <- list(x_no_fact,x_fact)
          x.inputs[sapply(x.inputs, is.null)] <- NULL
          x.inputs.val <- list(x_no_fact.val,x_fact.val)
          x.inputs.val[sapply(x.inputs.val, is.null)] <- NULL
        }

        print(layer$method_options$batch_size)

        if(layer$method_options$nfolds > 1) {
          history <- model %>%
            keras::fit(x=x.inputs, y=y, sample_weight = sample.w, epochs = layer$method_options$epochs,
                       batch_size = batch_size_tmp,
                       #validation_split = layer$method_options$validation_split,
                       validation_data = list(x.inputs.val,y.val,sample.w.val),
                       callbacks = list(earlystopping),
                       shuffle = layer$method_options$shuffle,
                       verbose = layer$method_options$verbose)
        }
        else {
          history <- model %>%
            keras::fit(x=x.inputs, y=y, sample_weight = sample.w, epochs = layer$method_options$epochs,
                       batch_size = batch_size_tmp,
                       validation_split = layer$method_options$validation_split,
                       #validation_data = list(x.inputs.val,y.val,sample.w.val),
                       callbacks = list(earlystopping),
                       shuffle = layer$method_options$shuffle,
                       verbose = layer$method_options$verbose)
        }

        score[k] <- ifelse(layer$method_options$bayesOpt.min, -min(history$metrics[[4]]), max(history$metrics[[4]]))

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

    if(is.null(layer$method_options$ae.hidden) &
       (('ae_hidden_1' %in% bounds_names) | ('ae_hidden_2' %in% bounds_names) | ('ae_hidden_3' %in% bounds_names)) ) {
      layer$method_options$ae.hidden <- c()
    }

    if('ae_hidden_1' %in% bounds_names) layer$method_options$ae.hidden[1] <- getBestPars(optObj)$ae_hidden_1
    if('ae_hidden_2' %in% bounds_names) layer$method_options$ae.hidden[2] <- getBestPars(optObj)$ae_hidden_2
    if('ae_hidden_3' %in% bounds_names) layer$method_options$ae.hidden[3] <- getBestPars(optObj)$ae_hidden_3

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

    if('batch_size' %in% bounds_names)
      layer$method_options$batch_size <- getBestPars(optObj)$batch_size*layer$method_options$batch_size

  }

  x <- layer$x

  if(layer$method_options$use_embedding) {
    x_fact        <-layer$x_fact
    x_no_fact     <-layer$x_no_fact
    no_fact_var   <-layer$no_fact_var
    fact_var      <- layer$fact_var
  }

  y <- layer$y
  sample.w <- weights.vec.n

  if(!is.null(layer$method_options$ae.hidden)) {

    ae_arch <- def_ae_arch(inputs=inputs,
                           x=x,
                           ae.hidden=layer$method_options$ae.hidden,
                           ae.activation.hidden=layer$method_options$ae.activation.hidden)

    autoencoder <- keras_model(inputs, ae_arch$ae_output_l)

    model_en <- keras_model(inputs, ae_arch$ae_hidden_l[[length(layer$method_options$ae.hidden)]])

    # Autoencoder model
    summary(autoencoder)
    # Encoder model
    summary(model_en)

    autoencoder %>% compile(loss = 'mae', optimizer='adam')

    autoencoder %>% keras::fit(
      x = x,
      y = x,
      epochs = layer$method_options$epochs,
      batch_size = layer$method_options$batch_size,
      verbose = layer$method_options$verbose,
      shuffle = layer$method_options$shuffle
    )

    x <- model_en %>% predict(x)
    inputs <- layer_input(shape = c(ncol(x)), name='ae_input_layer')
    layer$x.encoded <- x
    layer$model_en <- model_en

  }

  def_inputs <- def_inputs_mlp(use_embedding=layer$method_options$use_embedding,
                               x=x,
                               no_fact_var=no_fact_var,
                               fact_var=fact_var,
                               x_fact=x_fact,
                               output_dim = layer$method_options$output_dim)

  dnn_arch <- def_dnn_arch(inputs=def_inputs$inputs,
                           batch_normalization=layer$method_options$batch_normalization,
                           hidden=layer$method_options$hidden,
                           activation.hidden=layer$method_options$activation.hidden,
                           dropout.hidden=layer$method_options$dropout.hidden,
                           family_for_init=layer$method_options$family_for_init,
                           label=label,
                           data=data,
                           activation.output=layer$method_options$activation.output,
                           x=x,
                           use_bias=layer$method_options$use_bias,
                           weights.vec=weights.vec.n,
                           x_fact = x_fact,
                           x_no_fact = x_no_fact,
                           output_dim = layer$method_options$output_dim)

  if(!is.null(layer$method_options$family_for_init)) layer$glm.hom <- dnn_arch$glm.hom

  if(!layer$method_options$use_embedding) {
    model <- keras_model(inputs = def_inputs$inputs, outputs = c(dnn_arch$output))
  }
  else {
    model <- keras_model(inputs = c(def_inputs$inputs_no_fact,
                                    def_inputs$input_layer_emb),
                         outputs = c(dnn_arch$output))
  }

  print(summary(model))

  model %>% compile(
    loss = layer$method_options$loss,
    optimizer = layer$method_options$optimizer,
    metrics = layer$method_options$metrics
  )

  if(layer$method_options$gridsearch_cv) keras::set_weights(model, init_weights[[best_j]])

  earlystopping <- callback_early_stopping(
    monitor = layer$method_options$monitor,
    patience = layer$method_options$patience,
    restore_best_weights = T)

  indVal <- caret::createDataPartition(y=y,p=layer$method_options$validation_split, list=FALSE)

  if(!layer$method_options$use_embedding) {
      x.val    <- x[indVal,]
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
  }

  y.val    <- layer$y[indVal]
  sample.w.val <- weights.vec.n[indVal]

  if(!layer$method_options$use_embedding) {
    x <- x[-indVal,]
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
  }

  y <- layer$y[-indVal]
  sample.w <- weights.vec.n[-indVal]

  if(!layer$method_options$use_embedding) {
    x.inputs <- list(x)
    x.inputs.val <- list(x.val)
  }
  else {
    x.inputs <- list(x_no_fact,x_fact)
    x.inputs[sapply(x.inputs, is.null)] <- NULL
    x.inputs.val <- list(x_no_fact.val,x_fact.val)
    x.inputs.val[sapply(x.inputs.val, is.null)] <- NULL
  }

  # if(!layer$method_options$use_embedding) {
  #   x.inputs <- list(x)
  # }
  # else {
  #   x.inputs <- list(x_no_fact,x_fact)
  #   x.inputs[sapply(x.inputs, is.null)] <- NULL
  # }

  #now <- Sys.time()
  #fn <- paste0("./tmp/dnn_best_weights_",format(now, "%Y%m%d_%H%M%S.hdf5"))

  #CBs <- callback_model_checkpoint(fn, monitor=layer$method_options$monitor, save_best_only = TRUE, save_weights_only = TRUE)

  layer$history <- model %>%
    keras::fit(x=x.inputs, y=y, sample_weight=sample.w, epochs = layer$method_options$epochs,
               batch_size = layer$method_options$batch_size,
               validation_data = list(x.inputs.val,y.val,sample.w.val),
               #validation_split = layer$method_options$validation_split,
               #callbacks = list(earlystopping, CBs),
               callbacks = list(earlystopping),
               shuffle = layer$method_options$shuffle,
               verbose = layer$method_options$verbose)

  #load_model_weights_hdf5(model, fn)

  if(is.null(layer$method_options$hidden) & layer$method_options$bias_regularization) {
    cat('_ Bias regularization was activated but since there is no hidden layer, it will be deactivated for you.\n')
    layer$method_options$bias_regularization <- FALSE
  }

  if(!layer$method_options$bias_regularization) {
    layer$fit <- model
  }
  else {
    # We keep track of the neural network (biased) model
    layer$fit.biased <- model
    # Source: Ferrario, Andrea and Noll, Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April 23, 2020), p.52
    glm.formula <- function(nb) {
      string <- "yy ~ X1 "
      if(nb>1) {for (i in 2:nb) {string <- paste(string,"+ X",i, sep="")}}
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

    zz        <- keras_model(inputs = model$input, outputs=get_layer(model,'last_hidden_layer_activation')$output)
    layer$zz  <- zz

    if(!layer$method_options$use_embedding) {
      x <- def_x$x
      x.inputs <- list(x)
    }
    else {
      x_fact        <- def_x$x_fact
      x_no_fact     <- def_x$x_no_fact
      x.inputs      <- list(x_no_fact,x_fact)
    }

    Zlearn    <- data.frame(zz %>% predict(x.inputs))
    names(Zlearn) <- paste0('X', 1:ncol(Zlearn))

    Zlearn$yy <- layer$y

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

    glm1 <- glm(as.formula(glm.formula(ncol(Zlearn)-1)), data=Zlearn, family=fam, weights = weights.vec.n)
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
                                           data_baked_bc <- bake(layer$data_recipe, new_data = x)
                                           if(ncol(data_baked_bc) == 1)
                                             data_baked_bc <- data_baked_bc %>% mutate(intercept = 1)

                                           def_x <- def_x_mlp(layer$method_options$use_embedding,
                                                              layer$method_options$embedding_var,
                                                              f,
                                                              x,
                                                              data_baked_bc,
                                                              layer$data_recipe,
                                                              label)

                                           if(!layer$method_options$use_embedding) {
                                             x.tmp <- def_x$x
                                             if(!is.null(layer$method_options$ae.hidden))
                                               x.tmp <- model_en %>% predict(x.tmp)
                                             x.inputs.tmp <- list(x.tmp)
                                           }
                                           else {
                                             x_fact.tmp         <- def_x$x_fact
                                             x_no_fact.tmp      <- def_x$x_no_fact
                                             x.inputs.tmp <- list(x_no_fact.tmp,x_fact.tmp)
                                             x.inputs.tmp[sapply(x.inputs.tmp, is.null)] <- NULL
                                           }

                                           if(layer$method_options$bias_regularization) {
                                             Zlearn.tmp   <- data.frame(layer$zz %>% predict(x.inputs.tmp))
                                             names(Zlearn.tmp) <- paste0('X', 1:ncol(Zlearn.tmp))
                                             sum(x[[layer$name]])/sum(predict(layer$fit, newdata = Zlearn.tmp, type = 'response'))
                                           }
                                           else {
                                             sum(x[[layer$name]])/sum(layer$fit %>% predict(x.inputs.tmp))
                                           }
                                        })

      if(length(layer$balance.correction) != length(levels(data$dev.year.fact))) {
        ind <- !(levels(data$dev.year.fact) %in% names(layer$balance.correction))
        layer$balance.correction[as.character(which(ind))]=1
        layer$balance.correction = layer$balance.correction[levels(data$dev.year.fact)]
      }

  }

  if(!layer$method_options$use_embedding) {
    if(!is.null(layer$method_options$ae.hidden)) {
      x <- data.frame(model_en %>% predict(def_x$x)) %>% as.matrix()
    }
    else {
      x <- def_x$x
    }
    x.inputs <- list(x)
  }
  else {
    x.inputs <- list(def_x$x_no_fact,def_x$x_fact)
    x.inputs[sapply(x.inputs, is.null)] <- NULL
  }

  if(layer$method_options$bias_regularization) {
    # Zlearn   <- data.frame(obj$zz %>% predict(x.inputs))
    # names(Zlearn) <- paste0('X', 1:ncol(Zlearn))
    # pred <- predict(obj$fit, newdata = Zlearn, type = 'response') %>% as.matrix()
    pred <- layer$fit$fitted.values
  }
  else {
    pred <- layer$fit %>% predict(x.inputs)
  }

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

