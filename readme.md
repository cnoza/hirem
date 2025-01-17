# The `hirem` package 

This is the source code for the `hirem` package, forked from Jonas Crevecoeur's original version and merged with Jens Robben's updates extending the package with simulation machines based on various scenarios. 

The following layers are supported in the orginial version:

* `layer_glm`: Estimates a layer using a generalized linear model
* `layer_gbm`: Estimates a layer using a gradient boosting model

Supported distributions are:

* `binomial`
* `gaussian`
* `gamma`

To use the `gamma` distribution in `layer_gbm` you require an experimental version of the package `gbm` that implements the `gamma` distribution. See
https://github.com/harrysouthworth/gbm.

The new layers implemented in this fork are:

* `layer_xgb`: Estimates a layer using an extreme gradient boosting model (`xgboost` package)
* `layer_dnn`: Estimates a layer using an deep feed-forward neural network model (`keras` package)
* `layer_cann`: Estimates a layer using a Combined Actuarial Neural Network (CANN) model (`keras` package)

## Installation
To install this fork of the `hirem` package from GitHub you will need `devtools`:

``` r
install.packages('devtools')
devtools::install_github('cnoza/hirem')
```

## Overview
The `hirem` package offers tools for implementing hierarchical reserving models, introduced in the paper [Crevecoeur, J., Antonio, K., A generalized reserving model: bridging the gap between pricing and individual reserving, (2019)](https://arxiv.org/abs/1910.12692).

The `hirem` package constructs hierarchical reserving models by sequentially adding new layers. An example of a three component hierarchical model:

``` r
require(hirem)
data("reserving_data")

model <- hirem(reserving_data %>% dplyr::filter(calendar_year <= 6)) %>%
  layer_glm('close', binomial(link = logit)) %>%
  layer_glm('payment', binomial(link = logit)) %>%
  layer_glm('size', Gamma(link = log),
            filter = function(data){data$payment == 1})
            
model <- fit(model,
             close = 'close ~ factor(development_year)',
             payment = 'payment ~ close + factor(development_year)',
             size = 'size ~ close + factor(development_year)')
            
```

## Examples

Examples of tested models can be found in the following files:  

* [Use of an Extreme Gradient Boosting model as layer in `hirem`](https://htmlpreview.github.io?https://github.com/cnoza/hirem/blob/master/Examples/Use-of-Extreme-Gradient-Boosting-models-as-layer-in-hirem.html)
* [Use of a deep feed-forward neural network model as layer in `hirem`](https://htmlpreview.github.io?https://github.com/cnoza/hirem/blob/master/Examples/Use-of-a-MultiLayer-Perceptron-model-as-layer-in-hirem.html)
* [Use of CANN model as layer in `hirem`](https://htmlpreview.github.io?https://github.com/cnoza/hirem/blob/master/Examples/Use-of-a-CANN-model-as-layer-in-hirem.html)

