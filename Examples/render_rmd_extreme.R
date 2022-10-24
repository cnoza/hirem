# Launch Rmd file for different seeds

for(seed in 12:15) {
  rmarkdown::render("./Examples/Extreme with seed param.Rmd",
                    output_file=paste0("./Examples/extreme/html-no-cv/extreme_seed_",seed,".html"),
                    output_format = "all")
  rm(list = setdiff(ls(), seed))
}



