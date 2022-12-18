# Launch Rmd file for different seeds

for(seed in 1:1) {
  rmarkdown::render("./Examples/change-settlement/Change settle with seed param.Rmd",
                    output_file=paste0("./Examples/change-settlement/html-no-cv/no-recov/run-1/change_settle_seed_",seed,".html"),
                    output_format = "all")
  rm(list = setdiff(ls(), seed))
}



