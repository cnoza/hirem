# Launch Rmd file for different seeds

for(seed in 1:1) {
  rmarkdown::render("./Examples/mixed/Mixed claims with seed param (recoveries).Rmd",
                    output_file=paste0("./Examples/mixed/html-no-cv/recov/run-1/mixed_claims_recov_seed_",seed,".html"),
                    output_format = "all")
  rm(list = setdiff(ls(), seed))
}



