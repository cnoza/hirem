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
                         family_for_init,
                         label,
                         data,
                         activation.output,
                         x,
                         use_bias) {

  if(batch_normalization)
    output <- inputs %>% layer_batch_normalization()


  if(!is.null(hidden)) {

    n <- length(hidden)

    for(i in seq(from = 1, to=n)) {
      if(i==1 & !batch_normalization) {
        output <- inputs %>% layer_dense(units = hidden[i],
                                         name = ifelse(i==n,'last_hidden_layer',paste0('hidden_layer_',i))) %>%
          layer_activation(activation = activation.hidden[i])
        if(!is.null(dropout.hidden))
          output <- output %>% layer_dropout(rate = dropout.hidden[i])
      }
      else {
        output <- output %>% layer_dense(units = hidden[i],
                                         name = ifelse(i==n,'last_hidden_layer',paste0('hidden_layer_',i))) %>%
          layer_activation(activation = activation.hidden[i])
        if(!is.null(dropout.hidden))
          output <- output %>% layer_dropout(rate = dropout.hidden[i])
      }
    }

  }

  # Initialization with the homogeneous model
  # See Ferrario, Andrea and Noll, Alexander and Wuthrich, Mario V., Insights from Inside Neural Networks (April 23, 2020).
  # Available at SSRN: https://ssrn.com/abstract=3226852 or http://dx.doi.org/10.2139/ssrn.3226852 p.29.
  if(!is.null(family_for_init)) {

    f.hom <- paste0(label, '~ 1')
    glm.hom <- glm(as.formula(f.hom),data = data, family = family_for_init)

    if(!batch_normalization & is.null(hidden))
      output <- inputs %>% layer_dense(units = 1, activation = activation.output,
                                       weights = list(array(0,dim=c(ifelse(!is.null(hidden),hidden[n],c(ncol(x))),1)),
                                                      array(glm.hom$coefficients[1], dim=c(1))),
                                       use_bias = use_bias,
                                       name = 'output_layer')
    else
      output <- output %>% layer_dense(units = 1, activation = activation.output,
                                       weights = list(array(0,dim=c(ifelse(!is.null(hidden),hidden[n],c(ncol(x))),1)),
                                                      array(glm.hom$coefficients[1], dim=c(1))),
                                       use_bias = use_bias,
                                       name = 'output_layer')
  }
  else {
    if(!batch_normalization & is.null(hidden))
      output <- inputs %>% layer_dense(units = 1, activation = activation.output,
                                       use_bias = use_bias,
                                       name = 'output_layer')
    else
      output <- output %>% layer_dense(units = 1, activation = activation.output,
                                       name = 'output_layer')
  }

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

  if(!is.null(hidden)) {

    n <- length(hidden)

    for(i in seq(from = 1, to=n)) {
      if(i==1 & !batch_normalization) {
        NNetwork <- inputs %>%
          layer_dense(units = hidden[i],
                      name = ifelse(i==n,'last_hidden_layer',paste0('hidden_layer_',i))) %>%
          layer_activation(activation = activation.hidden[i])
        if(!is.null(dropout.hidden))
          NNetwork <- NNetwork %>% layer_dropout(rate = dropout.hidden[i])
      }
      else {
        NNetwork <- NNetwork %>% layer_dense(units = hidden[i],
                                             name = ifelse(i==n,'last_hidden_layer',paste0('hidden_layer_',i))) %>%
          layer_activation(activation = activation.hidden[i])
        if(!is.null(dropout.hidden))
          NNetwork <- NNetwork %>% layer_dropout(rate = dropout.hidden[i])
      }
    }

  }

  if(!batch_normalization & is.null(hidden))
    NNetwork <- inputs %>% layer_dense(units = 1, activation = activation.output,
                                       use_bias = use_bias,
                                       name = 'output_layer')
  else
    NNetwork <- NNetwork %>% layer_dense(units = 1, activation = activation.output,
                                         use_bias = use_bias,
                                         name = 'output_layer')

  return(NNetwork)

}
