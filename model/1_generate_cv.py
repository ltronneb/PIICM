import torch
import math
import os
import pandas as pd

# Script for generating datasets for cross-validation



# Set seed for reproducibility
torch.random.manual_seed(48652)
# Set constants for number of cell lines and drug combinations
n_cell = 39  # Number of cell lines
n_drugs = 583  # Number of drugs

# First load in raw data.
outputs = torch.tensor(pd.read_csv("data/ONeil_GP_full.csv").values)
noise = torch.tensor(pd.read_csv("data/ONeil_noise_full.csv").values)
n, p = outputs.shape

########################################################################################################################
# Missing entire experiments
########################################################################################################################
# Create mask
k = int(0.2 * p)  # 20% of experiments for validation
mask = torch.full((n, p), False)
probs = torch.full([p], 1.0)
sample = torch.multinomial(probs, k, replacement=False)
mask[:, sample] = True

# Create datasets
validation = outputs.masked_fill(~mask, float('nan'))
validation_noise = noise.masked_fill(~mask, float('nan'))
training = outputs.masked_fill(mask, float('nan'))
training_noise = noise.masked_fill(mask, float('nan'))
# and print these to file
pd.DataFrame(validation.numpy()).to_csv(os.path.join("data", "setting2_validation.csv"),index=False)
pd.DataFrame(validation_noise.numpy()).to_csv(os.path.join("data", "setting2_validation_noise.csv"),index=False)
pd.DataFrame(training.numpy()).to_csv(os.path.join("data", "setting2_training.csv"),index=False)
pd.DataFrame(training_noise.numpy()).to_csv(os.path.join("data", "setting2_training_noise.csv"),index=False)

# Further split training into cv folds, use 5-fold CV here
n_folds = 5
n_train = p - k
n_fold = int(n_train / 5)
mask = mask.expand((5, -1, -1)).clone()  # Expand mask to 5 cv loops
# Update probs
probs[sample] = 0
for L in range(5):
    sample = torch.multinomial(probs, n_fold, replacement=False)
    mask[L, :, sample] = True
    probs[sample] = 0
    cv = training.clone().masked_fill(mask[L, :, :], float('nan'))
    cv_noise = training_noise.clone().masked_fill(mask[L, :, :], float('nan'))
    cv = pd.DataFrame(cv.numpy())
    cv_noise = pd.DataFrame(cv_noise.numpy())
    cv.to_csv(os.path.join("data", "setting2_cv" + str(L) + ".csv"),index=False)
    cv_noise.to_csv(os.path.join("data", "setting2_cv" + str(L) + "_noise.csv"),index=False)
