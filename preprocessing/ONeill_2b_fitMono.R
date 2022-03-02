##### Script for fitting each experiment in [1] using the bayesynergy library
# 
# Input: 
#         Individual_Experiments/cell_line : drug.csv
#         .csv file containing the single example experiment data

# Output:
#         Individual_Experiments/mono/raw/cell_line : drug.RData
#         .RData file containing each single experiment fitted with bayesynergy

library(rstan)
library(foreach)

dir.create(file.path("Individual_Experiments/mono"), showWarnings = FALSE)
dir.create(file.path("Individual_Experiments/mono/raw"), showWarnings = FALSE)

readData = function(filename){
  data = read.table(paste0("Individual_Experiments/mono/",filename),header=T,sep=";")
  n = nrow(data)
  y = data[,4]
  x = log10(data[,3])
  return(list(n=n,y=y,x=x))
}


# Set up cluster ## Uncomment for parallel processing
#cl = parallel::makeCluster(40,outfile="debug_runExperiments.txt",type="FORK")
#doParallel::registerDoParallel(cl=cl)
#foreach::getDoParRegistered()

stan_model = stan_model("no_interaction.stan")

files = list.files("Individual_Experiments/mono")
#foreach(i = 1:length(files),.packages="bayesynergy") %dopar% { ## Uncomment for parallel processing
foreach(i = 1:length(files),.packages="bayesynergy") %do% {
  data = readData(files[i])
  fit = rstan::sampling(stan_model,data)
  save(fit,file=paste0("Individual_experiments/mono/raw/",substr(files[i],1,nchar(files[i])-4),".RData"))
}
  




# References:
# [1]
# Jennifer O'Neil, Yair Benita, Igor Feldman, Melissa Chenard, Brian Roberts, Yaping Liu, Jing Li, Astrid Kral, Serguei Lejnine, Andrey Loboda, William Arthur, Razvan Cristescu, Brian B. Haines, Christopher Winter, Theresa Zhang, Andrew Bloecher and Stuart D. Shumway
# Mol Cancer Ther June 1 2016 (15) (6) 1155-1162; DOI: 10.1158/1535-7163.MCT-15-0843

