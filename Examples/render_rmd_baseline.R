# Launch Rmd file for different seeds

for(seed in 11:100) {
  rmarkdown::render("./Examples/Baseline with seed param.Rmd",
                    output_file=paste0("./Examples/baseline/baseline_seed_",seed,".html"),
                    output_format = "all")
  rm(list = setdiff(ls(), seed))
}



