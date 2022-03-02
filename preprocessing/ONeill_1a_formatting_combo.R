##### Script for pre-processing the dataset from [1]
# 
# Input: 
#         156849_1_supp_0_w2lh45.xlsx (single agent viabilities)
#         156849_1_supp_1_w2lrww.xls  (combination viabilities) 
#         N.B. This file needs to be opened and resaved as xlsx
#         Both of these available as supplementary material for [1]

# Output:
#         Individual_Experiments/cell_line : drug_A + drug_B.csv
#         .csv file containing the single example experiment data


# Use reshape2 for preprocessing, readxl for input
library(reshape2)
library(readxl)

mono = read_excel("156849_1_supp_0_w2lh45.xlsx",col_types=c("numeric","text","text","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric"))
combo = read_excel("156849_1_supp_1_w2lrww.xlsx",col_types=c("numeric","text","text","numeric","text","numeric","text","numeric","numeric","numeric","numeric","numeric","numeric"))


# First, create a unique ID of the experiment
combo$ID = as.factor(paste(combo$cell_line,":",combo$combination_name))
ID = as.factor(unique(paste(combo$cell_line,":",combo$combination_name)))
# Now, for each unique ID, we need to pick out the relevant combination
# and monotherapy experiments

# Create empty vessel
ONeil = data.frame("BatchID"=NA,"ExperimentID"=NA,"cell_line"=NA,"combination_name"=NA,
                   "drugA_name"=NA,"drugA.Conc..µM."=NA,"drugB_name"=NA,
                   "drugB.Conc..µM."=NA,"viability"=NA,"mu.muMax"=NA,"X.X0"=NA)
kk = 1
dir.create(file.path("Individual_Experiments"), showWarnings = FALSE)
for (id in ID){
  comboSub = combo[which(combo$ID==id),]
  cellLine = as.character(comboSub$cell_line[1])
  drugA = as.character(comboSub$drugA_name[1])
  drugB = as.character(comboSub$drugB_name[1])
  # First subset to get all with the correct cellLine
  monoSub = mono[which(as.character(mono$cell_line)==cellLine),]
  # Now further down to get all with the correct drugs
  monoSubA = monoSub[which(as.character(monoSub$drug_name)==drugA),]
  monoSubB = monoSub[which(as.character(monoSub$drug_name)==drugB),]
  # Create new variables in old dataframes
  monoSubA$drugA_name = drugA
  monoSubA$`drugA Conc (µM)` = monoSubA$`Drug_concentration (µM)`
  monoSubA$drugB_name = drugB
  monoSubA$`drugB Conc (µM)` = 0
  monoSubA$combination_name = comboSub$combination_name[1]
  
  monoSubB$drugA_name = drugA
  monoSubB$`drugA Conc (µM)` = 0
  monoSubB$drugB_name = drugB
  monoSubB$`drugB Conc (µM)` = monoSubB$`Drug_concentration (µM)`
  monoSubB$combination_name = comboSub$combination_name[1]
  
  comboSub$viability5 = NA
  comboSub$viability6 = NA
  
  # Combining them all
  AllTogether = as.data.frame(rbind(comboSub[,names(comboSub)[c(1,2,3,4,5,6,7,8,9,10,11,15,16,12,13)]],
                                    monoSubA[,names(monoSubA)[c(1,2,13,14,15,16,17,5,6,7,8,9,10,11,12)]],
                                    monoSubB[,names(monoSubB)[c(1,2,13,14,15,16,17,5,6,7,8,9,10,11,12)]]))
  rownames(AllTogether) = NULL
  
  # Now for the reshape!
  reshaped = melt(AllTogether,id.vars = c("BatchID","cell_line","drugA_name","drugA Conc (µM)",
                                          "drugB_name", "drugB Conc (µM)","combination_name",
                                          "mu/muMax","X/X0"))
  reshaped$viability = reshaped$value
  reshaped = reshaped[,-c(10,11)]
  # Tagging on the experimentID
  reshaped$ExperimentID = id
  
  # And a final restacking with removal of NA
  reshaped = reshaped[-which(is.na(reshaped$viability)),c(1,11,2,7,3,4,5,6,10,8,9)]
  write.table(reshaped,file=paste0("Individual_Experiments/",as.character(id),".csv"),sep=";")
}


# References:
# [1]
# Jennifer O'Neil, Yair Benita, Igor Feldman, Melissa Chenard, Brian Roberts, Yaping Liu, Jing Li, Astrid Kral, Serguei Lejnine, Andrey Loboda, William Arthur, Razvan Cristescu, Brian B. Haines, Christopher Winter, Theresa Zhang, Andrew Bloecher and Stuart D. Shumway
# Mol Cancer Ther June 1 2016 (15) (6) 1155-1162; DOI: 10.1158/1535-7163.MCT-15-0843






