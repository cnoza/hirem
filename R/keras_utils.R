#' @export
def_ae_arch <- function(inputs,
                        x,
                        ae.hidden,
                        ae.activation.hidden) {

  nn <- length(ae.hidden)

  ae_hidden_l      <- list()
  ae_hidden_l[[1]] <- layer_dense(inputs,
                                  units = ae.hidden[1],
                                  activation = ae.activation.hidden[1],
                                  name='ae_encoding_layer_1')
  if(nn>1) {
    for(i in 2:nn) {
      ae_hidden_l[[i]] <- layer_dense(ae_hidden_l[[i-1]],
                                      units = ae.hidden[i],
                                      activation = ae.activation.hidden[i],
                                      name = ifelse(i==nn,'ae_bottleneck_layer',paste0('ae_encoding_layer_',i)))
    }
    for(i in 1:(nn-1)) {
      ae_hidden_l[[nn+i]] <- layer_dense(ae_hidden_l[[nn+i-1]],
                                         units = ae.hidden[nn-i],
                                         activation = ae.activation.hidden[nn-i],
                                         name = paste0('ae_decoding_layer_',i))
    }
  }

  ae_output_l <- layer_dense(ae_hidden_l[[(2*nn-1)]], ncol(x), name='ae_output_layer')

  return(list(ae_hidden_l=ae_hidden_l,ae_output_l=ae_output_l))

}

#' @export
def_mlp_arch <- function(inputs,
                         batch_normalization,
                         hidden,
                         activation.hidden,
                         dropout.hidden,
                         family_for_init=NULL,
                         label,
                         data,
                         activation.output,
                         x,
                         use_bias,
                         weights.vec) {

  if(batch_normalization)
    output <- inputs %>% layer_batch_normalization()
  else
    output <- inputs



  if(!is.null(hidden)) {

    n <- length(hidden)

    for(i in seq(from = 1, to=n)) {

        output <- output %>%
          layer_dense(units = hidden[i], name = ifelse(i==n,'last_hidden_layer',paste0('hidden_layer_',i))) %>%
          layer_activation(activation = activation.hidden[i], name = ifelse(i==n,'last_hidden_layer_activation',paste0('hidden_layer_activation_',i)))

        if(!is.null(dropout.hidden))
          output <- output %>% layer_dropout(rate = dropout.hidden[i])

    }

  }

  # Initialization with the homogeneous model
  # See Ferrario, Andrea and Noll, Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April 23, 2020).
  # Available at SSRN: https://ssrn.com/abstract=3226852 or http://dx.doi.org/10.2139/ssrn.3226852 p.29.
  if(!is.null(family_for_init)) {

    f.hom <- paste0(label, '~ 1')
    glm.hom <- glm(as.formula(f.hom),data = data, family = family_for_init, weights = weights.vec)

    output <- output %>%
      layer_dense(units = 1, activation = activation.output,
                  weights = list(array(0,dim=c(ifelse(!is.null(hidden),hidden[n],c(ncol(x))),1)),
                                 array(glm.hom$coefficients[1], dim=c(1))),
                  use_bias = use_bias,
                  name = 'output_layer')
  }
  else {
      output <- output %>% layer_dense(units = 1, activation = activation.output,
                                       use_bias = use_bias,
                                       name = 'output_layer')
  }

  if(is.null(family_for_init)) glm.hom <- NULL

  return(list(output=output,glm.hom=glm.hom))

}

#' @export
def_NN_arch <- function(inputs,
                        batch_normalization,
                        hidden,
                        activation.hidden,
                        dropout.hidden,
                        activation.output,
                        use_bias) {

  if(batch_normalization)
    NNetwork <- inputs %>% layer_batch_normalization()
  else
    NNetwork <- inputs

  if(!is.null(hidden)) {

    n <- length(hidden)

    for(i in seq(from = 1, to=n)) {

        NNetwork <- NNetwork %>%
          layer_dense(units = hidden[i],name = ifelse(i==n,'last_hidden_layer',paste0('hidden_layer_',i))) %>%
          layer_activation(activation = activation.hidden[i], name=ifelse(i==n,'last_hidden_layer_activation',paste0('hidden_layer_activation_',i)))

        if(!is.null(dropout.hidden))
          NNetwork <- NNetwork %>% layer_dropout(rate = dropout.hidden[i])

    }

  }

  NNetwork <- NNetwork %>% layer_dense(units = 1, activation = activation.output, use_bias = use_bias, name = 'output_layer_NN')

  return(NNetwork)

}

#' @export
def_x_mlp <- function(use_embedding,
                  f,
                  data,
                  data_baked,
                  label) {

  x <- select(data_baked,-as.name(label)) %>% as.matrix()

  if(use_embedding) {

    covariates      <- attr(terms(f),'term.labels')

    fact_var        <- covariates[covariates %in% names(data)[sapply(data, is.factor)]]
    no_fact_var     <- covariates[!(covariates %in% names(data)[sapply(data, is.factor)])]

    x_fact <- list()
    for(i in 1:length(fact_var)) {
      x_fact[[i]] <- select(data_baked,fact_var[i]) %>% as.matrix() %>% as.numeric()
      x_fact[[i]] <- x_fact[[i]]-1 # linked to issue with input_dim for embedding in keras
    }
    x_no_fact  <- select(data_baked,-as.name(label),-starts_with(fact_var)) %>% as.matrix()

    list_x <- list(x_fact=x_fact,
                   x_no_fact=x_no_fact,
                   fact_var=fact_var,
                   no_fact_var=no_fact_var,
                   x=x)

  }
  else {

    list_x <- list(x=x)

  }

  return(list_x)

}

#' @export
def_x <- function(use_embedding,
                  f,
                  f.glm,
                  data,
                  data_baked,
                  data_baked.glm,
                  label) {

  if(use_embedding) {

    covariates      <- attr(terms(f),'term.labels')
    covariates.glm  <- attr(terms(f.glm),'term.labels')

    fact_var        <- covariates[covariates %in% names(data)[sapply(data, is.factor)]]
    no_fact_var     <- covariates[!(covariates %in% names(data)[sapply(data, is.factor)])]
    fact_var.glm    <- covariates.glm[covariates.glm %in% names(data)[sapply(data, is.factor)]]
    no_fact_var.glm <- covariates.glm[!(covariates.glm %in% names(data)[sapply(data, is.factor)])]

    x_fact <- list()
    for(i in 1:length(fact_var)) {
      x_fact[[i]] <- select(data_baked,fact_var[i]) %>% as.matrix() %>% as.numeric()
      x_fact[[i]] <- x_fact[[i]]-1 # linked to issue with input_dim for embedding in keras
    }
    x_no_fact  <- select(data_baked,-as.name(label),-starts_with(fact_var)) %>% as.matrix()

    x_fact.glm <- list()
    for(i in 1:length(fact_var.glm)) {
      x_fact.glm[[i]] <- select(data_baked.glm,fact_var.glm[i]) %>% as.matrix() %>% as.numeric()
      x_fact.glm[[i]] <- x_fact.glm[[i]]-1 # linked to issue with input_dim for embedding in keras
    }
    x_no_fact.glm <- select(data_baked.glm,-as.name(label),-starts_with(fact_var.glm)) %>% as.matrix()

    list_x <- list(x_fact=x_fact,
                   x_no_fact=x_no_fact,
                   x_fact.glm=x_fact.glm,
                   x_no_fact.glm=x_no_fact.glm,
                   fact_var=fact_var,
                   no_fact_var=no_fact_var,
                   fact_var.glm=fact_var.glm,
                   no_fact_var.glm=no_fact_var.glm)

  }
  else {
    x           <- select(data_baked,-as.name(label)) %>% as.matrix()
    x.glm       <- select(data_baked.glm,-as.name(label)) %>% as.matrix()

    list_x <- list(x=x,x.glm=x.glm)

  }

  return(list_x)

}

#' @export
def_inputs_mlp <- function(use_embedding,
                           x,
                           no_fact_var = NULL,
                           fact_var = NULL,
                           x_fact = NULL,
                           output_dim = 1) {

  if(!use_embedding) {
    inputs     <- layer_input(shape = c(ncol(x)), name = 'input_layer_nn')

    list_inputs <- list(inputs=inputs)
  }
  else {

    inputs_no_fact      <- layer_input(shape = c(length(no_fact_var)), name = 'input_layer_no_fact')

    embedded_layer   <- list()
    input_layer_emb  <- list()

    for(i in 1:length(fact_var)) {

      input_layer_emb[[i]] <- layer_input(shape=c(1), dtype='int32', name = paste0('input_layer_',fact_var[i]))
      embedded_layer[[i]]  <- input_layer_emb[[i]] %>%
        layer_embedding(input_dim = max(x_fact[[i]])+1, output_dim = output_dim,
                        input_length = 1, name = paste0('embedding_layer_',fact_var[i]))

      embedded_layer[[i]] <- embedded_layer[[i]] %>%
        layer_flatten(name = paste0('embedding_layer_',fact_var[i],'_flat'))

    }

    inputs <- c(inputs_no_fact,embedded_layer) %>% layer_concatenate()

    list_inputs <- list(inputs=inputs,
                        inputs_no_fact=inputs_no_fact,
                        input_layer_emb=input_layer_emb)

  }

  return(list_inputs)

}

#' @export
def_inputs <- function(use_embedding,
                       x,
                       model.glm,
                       no_fact_var = NULL,
                       no_fact_var.glm = NULL,
                       fact_var.glm = NULL,
                       fact_var = NULL,
                       x_fact = NULL,
                       x_fact.glm = NULL) {

  if(!use_embedding) {
    inputs     <- layer_input(shape = c(ncol(x)), name = 'input_layer_nn')
    inputs.glm <- layer_input(shape = c(length(model.glm$coefficients)-1), name = 'input_layer_glm')

    list_inputs <- list(inputs.glm=inputs.glm,
                        inputs=inputs)
  }
  else {

    inputs_no_fact      <- layer_input(shape = c(length(no_fact_var)), name = 'input_layer_no_fact')
    inputs_no_fact.glm  <- layer_input(shape = c(length(no_fact_var.glm)), name = 'input_layer_no_fact_glm')

    model_coefficients <- model.glm$coefficients %>% as_tibble() %>% data.table::transpose()
    names(model_coefficients) <- attr(model.glm$coefficients,'names')

    embedded_layer.glm   <- list()
    input_layer_emb.glm  <- list()
    beta.no_fact_var.glm <- model_coefficients %>% select(!starts_with(fact_var.glm)) %>% as.numeric()
    beta.fact_var.glm    <- list()
    for(i in 1:length(fact_var.glm)) {
      beta.fact_var.glm[[i]]   <- model_coefficients %>% select(starts_with(fact_var.glm[i])) %>% as.numeric()
      input_layer_emb.glm[[i]] <- layer_input(shape=c(1), dtype='int32', name = paste0('input_layer_',fact_var.glm[i],'_glm'))
      embedded_layer.glm[[i]]  <- input_layer_emb.glm[[i]] %>%
        layer_embedding(input_dim = max(x_fact.glm[[i]])+1, output_dim = 1, trainable = FALSE,
                        input_length = 1, name = paste0('embedding_layer_',fact_var.glm[i],'_glm')
                        ,weights = list(array(c(beta.no_fact_var.glm[1],beta.fact_var.glm[[i]]),
                        dim=c(length(beta.fact_var.glm[[i]])+1,1)))
        ) %>%
        layer_flatten(name = paste0('embedding_layer_',fact_var.glm[i],'_flat_glm'))
    }

    inputs.glm <- c(inputs_no_fact.glm,embedded_layer.glm) %>% layer_concatenate()

    embedded_layer   <- list()
    input_layer_emb  <- list()
    #beta.no_fact_var <- model_coefficients %>% select(!starts_with(fact_var)) %>% as.numeric()
    #beta.fact_var    <- list()
    for(i in 1:length(fact_var)) {

      input_layer_emb[[i]] <- layer_input(shape=c(1), dtype='int32', name = paste0('input_layer_',fact_var[i]))
      embedded_layer[[i]]  <- input_layer_emb[[i]] %>%
        layer_embedding(input_dim = max(x_fact[[i]])+1, output_dim = 1,
                        input_length = 1, name = paste0('embedding_layer_',fact_var[i]))

      embedded_layer[[i]] <- embedded_layer[[i]] %>%
        layer_flatten(name = paste0('embedding_layer_',fact_var[i],'_flat'))

    }

    inputs <- c(inputs_no_fact,embedded_layer) %>% layer_concatenate()

    list_inputs <- list(inputs.glm=inputs.glm,
                        inputs=inputs,
                        inputs_no_fact=inputs_no_fact,
                        inputs_no_fact.glm=inputs_no_fact.glm,
                        input_layer_emb=input_layer_emb,
                        input_layer_emb.glm=input_layer_emb.glm,
                        beta.no_fact_var.glm=beta.no_fact_var.glm)

  }

  return(list_inputs)

}
