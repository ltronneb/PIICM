##### Script for setting up the required Tensors for input to PIICM, using preprocessed data from [1]
# 
# Input: 
#         Individual_Experiments/postPred/cell_line : drugA + drugB.csv
#         .csv file containing the single example experiment data, sampled from posterior predictive

# Output:
#    data/ONeil_concentrations.csv # The common grid of drug concentratiosn
#    data/ONeil_GP_full.csv" # Evaluations of the latent GP for each experiment
#    data/ONeil_f_full.csv" # Evaluations of the dose-response for each experiment
#    data/ONeil_p0_full.csv"# Evaluations of the non-interaction surface for each experiment
#    data/ONeil_noise_full.csv # Noise measurements for the latent GP
#    data/ONeil_f_noise_full.csv # Noise measurements for the dose-response
#    data/ONeil_b1_full.csv # Parameters used to transform latent GP to Interaction surface
#    data/ONeil_b2_full.csv  # Parameters used to transform latent GP to Interaction surface


# Using pandas to read in the initial csv files
import pandas as pd
import numpy as np
import re
import torch
# listdir for listing files
from os import listdir

# List all the csv files
files = listdir("preprocessing/Individual_Experiments/postPred/")

# Loop over all files and collect required information to set up the data properly
concentrations = None
drugs = []
combinations = []
celllines = []
for i in range(len(files)):
    # Read in the data
    data = pd.read_csv("preprocessing/Individual_Experiments/postPred/" + files[i], sep=";", header=0)
    data.columns = data.columns.str.replace('.', '')
    ### Get unique drug concentrations
    # Pull out unique drug concentrations
    x1 = data.DrugAconc.unique()
    x2 = data.DrugBconc.unique()
    x = np.unique(np.concatenate((x1, x2)))
    if concentrations is None:
        concentrations = x
    else:
        concentrations = np.unique(np.concatenate((concentrations, x)))

    ## Get cell line name
    result = re.search("^(.*?):", files[i])
    cellLine = result.group(1).strip()
    if celllines.count(cellLine) == 0:
        celllines.append(cellLine)
    ## Get drug names
    result = re.search(":(.+?)&", files[i])
    drugA = result.group(1).strip()
    result = re.search("&(.+?).csv", files[i])
    drugB = result.group(1).strip()
    if drugs.count(drugA) == 0:
        drugs.append(drugA)
    if drugs.count(drugB) == 0:
        drugs.append(drugB)
    combination = sorted([drugA, drugB], key=str.casefold)
    if combinations.count(combination) == 0:
        combinations.append(combination)

# Sort everything alphabetically
concentrations.sort()
celllines.sort()
drugs.sort()
combinations.sort(key=lambda x: x[0].lower())
print("num cell lines: ", len(celllines))
print("num drugs: ", len(drugs))
print("num combinations", len(combinations))
print("num unique concentrations", len(concentrations))

# Create expanded grid
x = torch.linspace(0,1,10)
y = torch.linspace(0,1,10)
[X,Y] = np.meshgrid(x,y)
train_X = X.reshape(-1)
train_Y = Y.reshape(-1)
train_inputs = torch.stack([
    torch.tensor(train_X),
    torch.tensor(train_Y)
],-1)

# Okay, so now create large train_output tensor containing and loop through every file
# Create empty tensors, filled with nans
train_output = torch.full((train_inputs.shape[0],len(combinations)*len(celllines)),np.nan)
train_noise = torch.full((train_inputs.shape[0],len(combinations)*len(celllines)),np.nan)
train_f = torch.full((train_inputs.shape[0],len(combinations)*len(celllines)),np.nan)
train_f_noise = torch.full((train_inputs.shape[0],len(combinations)*len(celllines)),np.nan)
train_p0 = torch.full((train_inputs.shape[0],len(combinations)*len(celllines)),np.nan)
train_b1 = torch.full((train_inputs.shape[0],len(combinations)*len(celllines)),np.nan)
train_b2 = torch.full((train_inputs.shape[0],len(combinations)*len(celllines)),np.nan)


for i in range(len(files)):
    # Read in data
    data = pd.read_csv("preprocessing/Individual_Experiments/postPred/"+files[i], sep=";", header=0)
    data.columns = data.columns.str.replace('.', '')
    ## Get drug names
    result = re.search(":(.+?)&",files[i])
    drugA = result.group(1).strip()
    result = re.search("&(.+?).csv",files[i])
    drugB = result.group(1).strip()
    # Make combination
    combination = sorted([drugA,drugB],key=str.casefold)
    # Figure out which combination index this corresponds to
    comb_index = combinations.index(combination)
    ## Get cell line name
    result = re.search("^(.*?):",files[i])
    cellLine = result.group(1).strip()
    # Figure out which cell line idex this corresponds to
    cell_index = celllines.index(cellLine)
    # Index in final tensor this corresponds to
    #index = len(combinations)*comb_index + cell_index
    index = len(combinations)*cell_index + comb_index # Tensor is blocked by cell lines, first 0:582 columns corresponds to cell line 1, next 583:1165 to cell line 2 and so on 
    print(i)
    ## Pull out unique drug concentrations
    # First check if drug A and drug B are flipped in the alphabetically sorted combination
    if combination.index(drugB) == 0:
        # If so, we swap the concentrations
        x2 = data.DrugAconc
        x1 = data.DrugBconc
    else:
        x1 = data.DrugAconc
        x2 = data.DrugBconc
    # Normalize to [0,1]
    X = torch.stack([torch.Tensor(x1),torch.Tensor(x2)],-1)
    # Now we fill in the grid
    for j in range(X.shape[0]):
        b = X[j,] # Current concentration
        min_idx = torch.norm(train_inputs - b.unsqueeze(0), dim=1).argmin() # Location of closest concentration
        train_output[min_idx,index] = data.GPMean[j]
        train_noise[min_idx,index] = data.GPVar[j]
        train_f[min_idx,index] = data.fMean[j]
        train_f_noise[min_idx,index] = data.fVAR[j]
        train_p0[min_idx,index] = data.p0Mean[j]
        train_b1[min_idx,index] = data.b1[j]
        train_b2[min_idx,index] = data.b2[j]

# Generate headers
headers = []
for i in range(len(celllines)):
    for j in range(len(combinations)):
        headers.append(celllines[i] + ":" + combinations[j][0] + "_" + combinations[j][1])

# Save these datasets
save_train_inputs = pd.DataFrame(train_inputs.numpy())
save_train_inputs.columns = ["X1","X2"]
save_train_output_f = pd.DataFrame(train_f.numpy())
save_train_output_f.columns = headers
save_train_output_p0 = pd.DataFrame(train_p0.numpy())
save_train_output_p0.columns = headers
save_train_output = pd.DataFrame(train_output.numpy())
save_train_output.columns = headers
save_train_noise = pd.DataFrame(train_noise.numpy())
save_train_noise.columns = headers
save_train_f_noise = pd.DataFrame(train_f_noise.numpy())
save_train_f_noise.columns = headers
save_train_b1 = pd.DataFrame(train_b1.numpy())
save_train_b1.columns = headers
save_train_b2 = pd.DataFrame(train_b2.numpy())
save_train_b2.columns = headers
save_train_inputs.to_csv("data/ONeil_concentrations.csv",index=False)
save_train_output.to_csv("data/ONeil_GP_full.csv",index=False)
save_train_output_f.to_csv("data/ONeil_f_full.csv",index=False)
save_train_output_p0.to_csv("data/ONeil_p0_full.csv",index=False)
save_train_noise.to_csv("data/ONeil_noise_full.csv",index=False)
save_train_f_noise.to_csv("data/ONeil_f_noise_full.csv",index=False)
save_train_b1.to_csv("data/ONeil_b1_full.csv",index=False)
save_train_b2.to_csv("data/ONeil_b2_full.csv",index=False)

# Also Save A Much smaller version with fewer combinations for benchmarking
n_drugcomb = 50
train_smaller = save_train_output.iloc[:, 0:(n_drugcomb*len(celllines))]
train_smaller_noise = save_train_noise.iloc[:, 0:(n_drugcomb*len(celllines))]
train_smaller_f = save_train_output_f.iloc[:, 0:(n_drugcomb*len(celllines))]
train_smaller_p0 = save_train_output_p0.iloc[:, 0:(n_drugcomb*len(celllines))]
train_smaller.to_csv("data/ONeil_GP_subset.csv",index=False)
train_smaller_noise.to_csv("data/ONeil_noise_subset.csv",index=False)
train_smaller_f.to_csv("data/ONeil_f_subset.csv",index=False)
train_smaller_p0.to_csv("data/ONeil_p0_subset.csv",index=False)
