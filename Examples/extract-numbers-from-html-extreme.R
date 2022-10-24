
library(dplyr)
library(rvest)

files <- list.files(path = "./Examples/extreme/html-no-cv/run-2/",pattern = ".html")

settlements <- list()
payments <- list()
sizes <- list()

for(i in seq_along(files)) {

  page <- read_html(paste0('./Examples/extreme/html-no-cv/run-2/',files[i]))

  # settlement

  settlements[i] <- page %>% html_node(xpath = '//*[@id="evaluating-the-predictive-performance-of-the-chain-ladder-model-and-the-hierarchical-glm-and-gbm"]/pre[2]/code/text()') %>% html_text()
  settlements[i] <- regmatches(settlements[[i]], gregexpr("[[:digit:]]+.[[:digit:]]+", settlements[[i]]))

  # payment

  payments[i] <- page %>% html_node(xpath = '//*[@id="evaluating-the-predictive-performance-of-the-chain-ladder-model-and-the-hierarchical-glm-and-gbm"]/pre[4]/code/text()') %>% html_text()
  payments[i] <- regmatches(payments[i], gregexpr("[[:digit:]]+.[[:digit:]]+", payments[i]))

  # size

  sizes[i] <- page %>% html_node(xpath = '//*[@id="evaluating-the-predictive-performance-of-the-chain-ladder-model-and-the-hierarchical-glm-and-gbm"]/pre[6]/code/text()') %>% html_text()
  sizes[i] <- regmatches(sizes[i], gregexpr("[[:digit:]]+.[[:digit:]]+", sizes[i]))

}

df.settle <- as.data.frame(do.call(rbind, settlements))
names(df.settle) <- c('Actual','CL','HGLM','HGBM','HXGB','HDNN','HCANN')

df.pay <- as.data.frame(do.call(rbind, payments))
names(df.pay) <- c('Actual','CL','HGLM','HGBM','HXGB','HDNN','HCANN')

df.size <- as.data.frame(do.call(rbind, sizes))
names(df.size) <- c('Actual','CL','HGLM','HGBM','HXGB','HDNN','HCANN')

write_csv(df.settle,'./Examples/extreme/html-no-cv/run-2/results-simulation-baseline-settlements.csv')
write_csv(df.pay,'./Examples/extreme/html-no-cv/run-2/results-simulation-baseline-payments.csv')
write_csv(df.size,'./Examples/extreme/html-no-cv/run-2/results-simulation-baseline-sizes.csv')


df.settle <- df.settle %>% mutate(error.cl=abs(as.numeric(Actual)-as.numeric(CL))/as.numeric(Actual))
df.settle <- df.settle %>% mutate(error.hglm=abs(as.numeric(Actual)-as.numeric(HGLM))/as.numeric(Actual))
df.settle <- df.settle %>% mutate(error.hgbm=abs(as.numeric(Actual)-as.numeric(HGBM))/as.numeric(Actual))
df.settle <- df.settle %>% mutate(error.hxgb=abs(as.numeric(Actual)-as.numeric(HXGB))/as.numeric(Actual))
df.settle <- df.settle %>% mutate(error.hdnn=abs(as.numeric(Actual)-as.numeric(HDNN))/as.numeric(Actual))
df.settle <- df.settle %>% mutate(error.hcann=abs(as.numeric(Actual)-as.numeric(HCANN))/as.numeric(Actual))

settl.error.cl <- sum(df.settle$error.cl)/100
settl.error.hglm <- sum(df.settle$error.hglm)/100
settl.error.hgbm <- sum(df.settle$error.hgbm)/100
settl.error.hxgb <- sum(df.settle$error.hxgb)/100
settl.error.hdnn <- sum(df.settle$error.hdnn)/100
settl.error.hcann <- sum(df.settle$error.hcann)/100

df.pay <- df.pay %>% mutate(error.cl=abs(as.numeric(Actual)-as.numeric(CL))/as.numeric(Actual))
df.pay <- df.pay %>% mutate(error.hglm=abs(as.numeric(Actual)-as.numeric(HGLM))/as.numeric(Actual))
df.pay <- df.pay %>% mutate(error.hgbm=abs(as.numeric(Actual)-as.numeric(HGBM))/as.numeric(Actual))
df.pay <- df.pay %>% mutate(error.hxgb=abs(as.numeric(Actual)-as.numeric(HXGB))/as.numeric(Actual))
df.pay <- df.pay %>% mutate(error.hdnn=abs(as.numeric(Actual)-as.numeric(HDNN))/as.numeric(Actual))
df.pay <- df.pay %>% mutate(error.hcann=abs(as.numeric(Actual)-as.numeric(HCANN))/as.numeric(Actual))

payment.error.cl <- sum(df.pay$error.cl)/100
payment.error.hglm <- sum(df.pay$error.hglm)/100
payment.error.hgbm <- sum(df.pay$error.hgbm)/100
payment.error.hxgb <- sum(df.pay$error.hxgb)/100
payment.error.hdnn <- sum(df.pay$error.hdnn)/100
payment.error.hcann <- sum(df.pay$error.hcann)/100

df.size <- df.size %>% mutate(error.cl=abs(as.numeric(Actual)-as.numeric(CL))/as.numeric(Actual))
df.size <- df.size %>% mutate(error.hglm=abs(as.numeric(Actual)-as.numeric(HGLM))/as.numeric(Actual))
df.size <- df.size %>% mutate(error.hgbm=abs(as.numeric(Actual)-as.numeric(HGBM))/as.numeric(Actual))
df.size <- df.size %>% mutate(error.hxgb=abs(as.numeric(Actual)-as.numeric(HXGB))/as.numeric(Actual))
df.size <- df.size %>% mutate(error.hdnn=abs(as.numeric(Actual)-as.numeric(HDNN))/as.numeric(Actual))
df.size <- df.size %>% mutate(error.hcann=abs(as.numeric(Actual)-as.numeric(HCANN))/as.numeric(Actual))

size.error.cl <- sum(df.size$error.cl)/100
size.error.hglm <- sum(df.size$error.hglm)/100
size.error.hgbm <- sum(df.size$error.hgbm)/100
size.error.hxgb <- sum(df.size$error.hxgb)/100
size.error.hdnn <- sum(df.size$error.hdnn)/100
size.error.hcann <- sum(df.size$error.hcann)/100

