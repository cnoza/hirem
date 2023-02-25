# Launch Rmd file for different seeds

for(seed in 1:5) {
  rmarkdown::render("./Examples/wang-wuthrich/Wang-Wuthrich with seed param (with recoveries 8).Rmd",
                    output_file=paste0("./Examples/wang-wuthrich/html-no-cv/wang-wuthrich-recov8_seed_",seed,".html"),
                    output_format = "all")
  rm(list = setdiff(ls(), seed))
}



