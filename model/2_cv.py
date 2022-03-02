import numpy as np
import pandas as pd
import os
import torch
import argparse
import runModel

"""
Script for fitting the model for to each of the 5 CV sets and saving the output

Arguments:
    ranksetting: int, the setting for the ranks of cell line and drug combination covariance matrices
"""
# Some constants
N_drugs = 583
N_celllines = 39

# Setting
setting = 2

# Create the parser
nparser = argparse.ArgumentParser(prog='cv',
                                  description='Program for fitting data to the 5 CV folds')
# Add the arguments
#nparser.add_argument('-setting', type=int, help='cv setting')
nparser.add_argument('-ranksetting', type=int, help='rank setting')
nparser.add_argument('-s', dest='symmetric', help='run with symmetries', action='store_true')
nparser.add_argument('-ns', dest='symmetric', help='run without symmetries', action='store_false')
nparser.set_defaults(symmetric=True,setting=2)

# Execute the parse_args() method
args = nparser.parse_args()
# Pull out arguments
#setting = args.setting
ranksetting = args.ranksetting
symmetric = args.symmetric

# Construct the ranks
#cellRanks = np.array([1, 3, 5, 10, 20])
cellRanks = np.array([39])
drugRanks = np.array([5, 50, 100, 150, 300])
[X, Y] = np.meshgrid(cellRanks, drugRanks)
cellRanks = X.reshape(-1)
drugRanks = Y.reshape(-1)

cellRank = cellRanks[ranksetting]
drugRank = drugRanks[ranksetting]

# First read in full training dataset
Y = torch.tensor(pd.read_csv(os.path.join("data", "setting"+str(setting)+"_training.csv")).values).float()
S = torch.tensor(pd.read_csv(os.path.join("data", "setting"+str(setting)+"_training_noise.csv")).values).float()
# Observations missing in the training data (i.e. validation set)
idx_train = Y.isnan()
# Create tensor to store predictions
pred = Y.clone()

for L in range(5):
    print("cv loop:", L)
    # Read in data
    # Inputs
    X = torch.tensor(pd.read_csv("data/ONeil_concentrations.csv").values).float()
    # Outputs
    y = torch.tensor(pd.read_csv(os.path.join("data", "setting"+str(setting)+"_cv" + str(L) + ".csv")).values).float()
    s = torch.tensor(pd.read_csv(os.path.join("data", "setting"+str(setting)+"_cv" + str(L) + "_noise.csv")).values).float()

    # Run the model
    mean = runModel.runModel(X, y, s, N_drugs, N_celllines, drugRank, cellRank, symmetric=symmetric,tol=1e-3,cg_tol=0.1,eval_cg_tol=0.05)
    #if symmetric:
    #    mean = mean[:, 0:int(N_drugs*N_celllines)]
    # Those missing in the cv fold:
    idx_cv = y.isnan()
    # Those missing only in the cv fold
    idx = (~idx_train)*(idx_cv)
    # Store in the prediction tensor
    pred[idx] = mean[idx]

print(pred)

# Observed
o = Y[~idx_train]
# Predicted
p = pred[~idx_train]
# Weights
w = 1/S[~idx_train]
# Normalize weights
w = w/(w.sum())

wRMSE = (w*(o-p).square()).sum().sqrt().item()
RMSE =  ((o-p).square()).mean().sqrt().item()

print("wRMSE =", wRMSE)
print("RMSE=", RMSE)


data = [[setting, drugRank, cellRank, wRMSE, RMSE, symmetric]]
df = pd.DataFrame(data, columns=['CV setting', 'Drug Rank', "Cell line Rank", "wRMSE", "RMSE", "Symmetric"])
df.to_csv(os.path.join("results","setting"+str(setting)+"_results.csv"), mode='a', header=not os.path.exists(os.path.join("results","setting"+str(setting)+"_results.csv")),index=False)

print("Script done!")
