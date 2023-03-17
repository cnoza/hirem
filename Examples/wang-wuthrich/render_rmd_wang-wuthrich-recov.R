# Launch Rmd file for different seeds

for(seed in 1:3) {
  rmarkdown::render("./Examples/wang-wuthrich/Wang-Wuthrich with seed param (with recoveries 10).Rmd",
                    output_file=paste0("./Examples/wang-wuthrich/html-no-cv/run1/wang-wuthrich-recov10_seed_",seed,".html"),
                    output_format = "all")
  rm(list = setdiff(ls(), seed))
}



