##### Script for pre-processing the dataset from [1]
# 
# Input: 
#         156849_1_supp_0_w2lh45.xlsx (single agent viabilities)
#         This file is available as supplementary material for [1]

# Output:
#         Individual_Experiments/mono/cell_line : drug.csv
#         .csv file containing the single example experiment data


# Use reshape2 for preprocessing, readxl for input

library(readxl)
library(tidyr)

dir.create(file.path("Individual_Experiments/mono"), showWarnings = FALSE)

data = read_xlsx("156849_1_supp_0_w2lh45.xlsx",na="NULL")
data$cell_line = as.factor(data$cell_line)
data$drug_name = as.factor(data$drug_name)

for (i in 1:length(levels(data$cell_line))){
  for (j in 1:length(levels(data$drug_name))){
    sub = data %>% 
      select("cell_line","drug_name","Drug_concentration (µM)",starts_with("viability")) %>%
      filter(`cell_line`==levels(data$cell_line)[i]) %>%
      filter(`drug_name`==levels(data$drug_name)[j]) %>%
      pivot_longer(cols=starts_with("viability"),values_drop_na=T) %>%
      select("cell_line", "drug_name", "Drug_concentration (µM)", "value") %>%
      rename("viability"="value")
    
    write.table(sub, file=paste0("Individual_Experiments/mono/",levels(data$cell_line)[i]," : ",levels(data$drug_name)[j],".csv"),row.names = F,sep=";")
  }
}


# References:
# [1]
# Jennifer O'Neil, Yair Benita, Igor Feldman, Melissa Chenard, Brian Roberts, Yaping Liu, Jing Li, Astrid Kral, Serguei Lejnine, Andrey Loboda, William Arthur, Razvan Cristescu, Brian B. Haines, Christopher Winter, Theresa Zhang, Andrew Bloecher and Stuart D. Shumway
# Mol Cancer Ther June 1 2016 (15) (6) 1155-1162; DOI: 10.1158/1535-7163.MCT-15-0843





