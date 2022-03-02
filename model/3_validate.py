import numpy as np
import pandas as pd
import os
import torch
import argparse
import runModel

"""
Script for running model on validation set

Arguments:
    ranksetting: int, the setting for the ranks of cell line and drug combination covariance matrices
"""
# Some constants
N_drugs = 583
N_celllines = 39
setting = 2

# Create the parser
nparser = argparse.ArgumentParser(prog='cv',
                                  description='Program for fitting data to the 5 CV folds')
# Add the arguments
nparser.add_argument('-s', dest='symmetric', help='run with symmetries', action='store_true')
nparser.add_argument('-ns', dest='symmetric', help='run without symmetries', action='store_false')
nparser.set_defaults(symmetric=True)

# Execute the parse_args() method
args = nparser.parse_args()
# Pull out arguments
symmetric = args.symmetric

# Set the ranks
cv_results = pd.read_csv(os.path.join("results","setting"+str(setting)+"_results.csv"))
idx = cv_results[["wRMSE"]].idxmin()

cellRank = int(cv_results.iloc[idx,2].to_numpy())
drugRank = int(cv_results.iloc[idx,1].to_numpy())

# First read in full training dataset
X = torch.tensor(pd.read_csv("data/ONeil_concentrations.csv").values).float()
Y = torch.tensor(pd.read_csv(os.path.join("data", "setting"+str(setting)+"_training.csv")).values).float()
S = torch.tensor(pd.read_csv(os.path.join("data", "setting"+str(setting)+"_training_noise.csv")).values).float()
#S = torch.zeros(Y.shape).float()
# Observations missing in the training data (i.e. validation set)
idx_train = Y.isnan()
 
# Fit model
mean = runModel.runModel(X, Y, S, N_drugs, N_celllines, drugRank, cellRank, symmetric=symmetric,tol=1e-3,cg_tol=0.01,eval_cg_tol=0.01,save_outputs=True,setting=setting)

# Read in validation dataset
V = torch.tensor(pd.read_csv(os.path.join("data", "setting"+str(setting)+"_validation.csv")).values).float()
Z = torch.tensor(pd.read_csv(os.path.join("data", "setting"+str(setting)+"_validation_noise.csv")).values).float()

# Observed
o = V[idx_train]
# Predicted
p = mean[idx_train]
# Weights
w = 1/Z[idx_train]
# Normalize weights
w = w/(w.sum())

wRMSE = (w*(o-p).square()).sum().sqrt().item()
RMSE =  ((o-p).square()).mean().sqrt().item()

print("wRMSE =", wRMSE)
print("RMSE=", RMSE)


data = [[setting, drugRank, cellRank, wRMSE, RMSE, symmetric]]
df = pd.DataFrame(data, columns=['setting', 'Drug Rank', "Cell line Rank", "wRMSE", "RMSE", "Symmetric"])
df.to_csv(os.path.join("results","validation_results.csv"), mode='a', header=not os.path.exists(os.path.join("results","validation_results.csv")),index=False)

print("Script done!")
