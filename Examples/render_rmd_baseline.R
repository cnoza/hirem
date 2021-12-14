# Launch Rmd file for different seeds

#install_github("petrelharp/templater")
library(templater)

for (seed in 1:2) {
  render_template("./Examples/Scenario testing on simulated portfolio (baseline scenario) parallel.Rmd", output=paste0("baseline_scenario_",seed,".html"))
}
