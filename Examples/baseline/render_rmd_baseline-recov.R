# Launch Rmd file for different seeds

for(seed in 1:1) {
  rmarkdown::render("./Examples/baseline/Baseline with seed param (with recoveries).Rmd",
                    output_file=paste0("./Examples/baseline/html-no-cv/recov/run-1/baseline_seed_recov_",seed,".html"),
                    output_format = "all")
  rm(list = setdiff(ls(), seed))
}

