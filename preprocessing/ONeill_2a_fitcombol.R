##### Script for fitting each experiment in [1] using the bayesynergy library
# 
# Input: 
#         Individual_Experiments/cell_line : drug_A + drug_B.csv
#         .csv file containing the single example experiment data

# Output:
#         Individual_Experiments/raw/cell_line : drug_A + drug_B.RData
#         .RData file containing each single experiment fitted with bayesynergy

library(bayesynergy)
library(foreach)

dir.create(file.path("Individual_Experiments/raw"), showWarnings = FALSE)

readData = function(filename){
  data = read.table(paste0("Individual_Experiments/raw",filename),header=F,sep=";",dec=".",skip=1)
  x = as.matrix(data[,c(7,9)])
  y = as.matrix(data[,10])
  return(list(y = y, x = x, drug_names = c(data[1,6],data[1,8]), experiment_ID = data[1,4]))
}


# Set up cluster ## Uncomment for parallel processing
#cl = parallel::makeCluster(40,outfile="debug_runExperiments.txt",type="FORK")
#doParallel::registerDoParallel(cl=cl)
#foreach::getDoParRegistered()

files = list.files("Individual_Experiments/")
#foreach(i = 1:length(files),.packages="bayesynergy") %dopar% { ## Uncomment for parallel processing
foreach(i = 1:length(files),.packages="bayesynergy") %do% {
  data = readData(files[i])
  #data$method="vb" ## For VB instead of full Bayes
  fit = do.call(bayesynergy,data)
  # Save raw fit to file
  save(fit,file=paste0("Individual_Experiments/raw/",substr(files[i],1,nchar(files[i])-4),".RData"))
}
  


# References:
# [1]
# Jennifer O'Neil, Yair Benita, Igor Feldman, Melissa Chenard, Brian Roberts, Yaping Liu, Jing Li, Astrid Kral, Serguei Lejnine, Andrey Loboda, William Arthur, Razvan Cristescu, Brian B. Haines, Christopher Winter, Theresa Zhang, Andrew Bloecher and Stuart D. Shumway
# Mol Cancer Ther June 1 2016 (15) (6) 1155-1162; DOI: 10.1158/1535-7163.MCT-15-0843



