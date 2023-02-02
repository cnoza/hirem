# Launch Rmd file for different seeds

for(seed in 1:1) {
  rmarkdown::render("./Examples/wang-wuthrich/Wang-Wuthrich with seed param (with recoveries 3).Rmd",
                    output_file=paste0("./Examples/wang-wuthrich/html-no-cv/wang-wuthrich-recov3_seed_",seed,".html"),
                    output_format = "all")
  rm(list = setdiff(ls(), seed))
}



