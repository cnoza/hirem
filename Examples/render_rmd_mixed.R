# Launch Rmd file for different seeds

for(seed in 1:1) {
  rmarkdown::render("./Examples/Mixed claims with seed param.Rmd",
                    output_file=paste0("./Examples/mixed/html-no-cv/mixed_claims_seed_",seed,".html"),
                    output_format = "all")
  rm(list = setdiff(ls(), seed))
}



