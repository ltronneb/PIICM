# Some initial imports needed to set up everything
import math
import torch
import gpytorch
import numpy as np
import pandas as pd
import argparse
import os

"""
Script for running the drug response model and imputing missing values given an
input dataset and ranks for drug and cell line covariance.

Arguments:
    -d Path to data
    -n Path to noise data
    -p Output path, to save predictions 
    -s Run symmetric version of the model
    -ns Run non-symmetric version of the model
    -rd Rank of drug covariance
    -rc Rank of cell line covariance
    -maxiter Max iterations for Adam optimizer (default: 200)
    -tol Tolerance of convergence (default: 1e-3)
"""

def runModel(X, y, s, N_combinations, N_cellLines, drugRank, cellRank, symmetric=True, maxiter=200, tol=1e-3,cg_tol=0.01,eval_cg_tol=0.01,save_outputs=False,setting=0):
    # If we are in the symmetric setting, expand the input dimension
    if symmetric:
        # Double the drug rank
        drugRank = 2*drugRank
        # Set half tolerance
        tol = tol / 2
        # Expand the input dimension
        n, p = y.shape
        p = 2*p
        tmp = torch.full([n,p],float('nan'))
        tmp_noise = torch.full([n,p],float('nan'))
        for i in range(N_cellLines):
            orig = y[:,(i*N_combinations):(i*N_combinations+N_combinations)]
            orig_noise = s[:,(i*N_combinations):(i*N_combinations+N_combinations)]
            new = torch.full(orig.shape,float('nan'))
            new_noise = torch.full(orig.shape,float('nan'))
            combined = torch.cat([orig,new],-1)
            combined_noise = torch.cat([orig_noise,new_noise],-1)
            tmp[:,(i*(2*N_combinations)):(i*(2*N_combinations)+2*N_combinations)] = combined
            tmp_noise[:,(i*(2*N_combinations)):(i*(2*N_combinations)+2*N_combinations)] = combined_noise
        y = tmp
        s = tmp_noise

    # Fill in missing values with dummy variables and mask
    # Pad with missing values and large noise
    mVAL = -999
    mVAR = 99999999999999
    missing_idx = y.isnan()
    N_missing = missing_idx.sum()
    # Train output is given a fixed dummy values
    y = y.masked_fill(missing_idx, mVAL)
    # Correspondingly these are given a large noise
    s = s.masked_fill(missing_idx, mVAR)

    # For CUDA support
    if torch.cuda.is_available():
        print("CUDA available!")
        X, y, s = X.cuda(), y.cuda(), s.cuda()
    else:
        print("CUDA not available!")

    # Define some constants
    if symmetric:
        N_tasks = int(2 * N_combinations * N_cellLines)
    else:
        N_tasks = int(N_combinations * N_cellLines)
    N_total = y.numel()
    N_obs = int(N_total - N_missing)
    # Adjust cg_tol to sparsely observed data
    cg_tol = cg_tol * (N_obs/N_total)
    eval_cg_tol = eval_cg_tol * (N_obs / N_total)

    # Here now set up the model
    class MultitaskGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ZeroMean(), num_tasks=N_tasks
            )
            self.covar_module = gpytorch.kernels.DrugResponseKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
                num_combinations=N_combinations, num_cell_lines=N_cellLines,
                drug_rank=drugRank, cell_linerank=cellRank,
                symmetric=symmetric
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.FixedNoiseMultitaskGaussianLikelihood(num_tasks=N_tasks,
                                                                            noise=s)
    model = MultitaskGPModel(X, y, likelihood)

    # Initialize cell line covariance                                                                                                                                                                                                                  
    # Initialize some parameters                                                                                                                                                                                                                        
    # First create covar_matrix for cell lines                                                                                                                                                                                                         
    #M = torch.eye(N_cellLines).float()
    #for L in range(6):
    #    idx = cells.index[cells["Tissue"] == L].tolist()
    #    for i in range(len(idx)):
    #        for j in range(len(idx)):
    #            M[idx[i],idx[j]] = 1.0
    #            M[idx[j],idx[i]] = 1.0
    # Now we decomp this                                                                                                                                                                                                                               
    #eig = torch.linalg.eigh(M)
    #values = eig[0]
    #values = values.clamp_min(0.0)
    #vectors = eig[1]
    # Root the values                                                                                                                                                                                                                                  
    #values = torch.sqrt(values)
    #M.sqrt = vectors.matmul(torch.diag(values)).matmul(vectors.inverse()) + 0.1*torch.randn(M.shape)

    # Initialization
    #model.initialize(**{'covar_module.cellline_covar_module.covar_factor':M.sqrt})

    # For CUDA support
    if torch.cuda.is_available():
        print("CUDA available!")
        model = model.cuda()
        likelihood = likelihood.cuda()

    # Training
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    # Loss
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # Keeping track of loss
    m = []
    diff = 10
    with gpytorch.settings.verbose_linalg(True), gpytorch.settings.max_preconditioner_size(0):
        with gpytorch.settings.use_eigvalsh(True), gpytorch.settings.max_cg_iterations(2000):
            with gpytorch.settings.cg_tolerance(cg_tol):
                for i in range(maxiter):
                    optimizer.zero_grad()
                    output = model(X)
                    loss = -mll(output, y)
                    loss.backward()
                    optimizer.step()
                    m.append(loss.item())
                    if i > 0:
                        diff = abs(m[i] - m[i - 1])
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f  loss difference: %.4f' % (
                        i + 1, maxiter, loss.item(),
                        model.covar_module.data_covar_module.base_kernel.lengthscale.item(),
                        model.likelihood.global_noise.item(),
                        diff
                    ))
                    if diff < tol:
                        print('Model converged!')
                        break

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Put stuff back on cpu because of memory issues
    model = model.cpu()
    likelihood = likelihood.cpu()
    X = X.cpu()
    y = y.cpu()
    s = s.cpu()
    
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.max_cg_iterations(4000), gpytorch.settings.eval_cg_tolerance(eval_cg_tol), gpytorch.settings.verbose_linalg(True), gpytorch.settings.skip_posterior_variances(True):
        predictions = likelihood(model(X))
        mean = predictions.mean

    if symmetric:
        n,p = mean.shape
        p = int(p/2)
        tmp = torch.full([n,p], float('nan'))
        for i in range(N_cellLines):
            tmp[:,(i*N_combinations):(i*N_combinations+N_combinations)] = mean[:,(i*(2*N_combinations)):(i*(2*N_combinations)+N_combinations)]
        mean = tmp

    # Save outputs
    if save_outputs:
        drug_covar = pd.DataFrame(model.covar_module.drugcombo_covar_module.covar_matrix.evaluate().detach().numpy())
        cell_covar = pd.DataFrame(model.covar_module.cellline_covar_module.covar_matrix.evaluate().detach().numpy())
        predicted = pd.DataFrame(mean.detach().numpy())
        drug_covar.to_csv(os.path.join("results","validation_drug_covar-setting="+str(setting)+"-rd="+str(drugRank)+"-rc="+str(cellRank)+".csv"),index=False)
        cell_covar.to_csv(os.path.join("results","validation_cell_covar-setting="+str(setting)+"-rd="+str(drugRank)+"-rc="+str(cellRank)+".csv"),index=False)
        predicted.to_csv(os.path.join("results","validation_predicted-setting="+str(setting)+"-rd="+str(drugRank)+"-rc="+str(cellRank)+".csv"),index=False)
    return mean

if __name__ == '__main__':
    # Create the parser
    nparser = argparse.ArgumentParser(prog='runModel',
                                      description='Main program for drug response prediction')
    # Add the arguments
    nparser.add_argument('-d', type=str, help='path to data')
    nparser.add_argument('-n', type=str, help='path to noise data')
    nparser.add_argument('-p', type=str, help='output path')
    nparser.add_argument('-s', dest='symmetric', help='run with symmetries', action='store_true')
    nparser.add_argument('-ns', dest='symmetric', help='run without symmetries', action='store_false')
    nparser.set_defaults(symmetric=True)
    nparser.add_argument('-nd', type=int, help='number of drug combinations')
    nparser.add_argument('-nc', type=int, help='number of cell lines')
    nparser.add_argument('-rd', type=int, help='rank of the drug covariance matrix')
    nparser.add_argument('-rc', type=int, help='rank of the cell line covariance matrix')
    nparser.add_argument('-maxiter', default=200, type=int, help='maximum iterations')
    nparser.add_argument('-tol', default=1e-3, type=float, help='tolerance for convergence')

    # Execute the parse_args() method
    args = nparser.parse_args()

    # Get the arguments
    inputdata = args.d
    inputnoise = args.n
    output = args.p
    symmetric = args.symmetric
    N_combinations = args.nd
    N_cellLines = args.nc
    drugRank = args.rd
    cellRank = args.rc
    maxiter = args.maxiter
    tol = args.tol

    # Read in data and split into train/test
    # Inputs
    X = torch.tensor(pd.read_csv("data/ONeil_concentrations.csv").values).float()
    y = torch.tensor(pd.read_csv(inputdata).values).float()
    s = torch.tensor(pd.read_csv(inputnoise).values).float()
    mean = runModel(X, y, s, N_combinations, N_cellLines, drugRank, cellRank, symmetric, maxiter, tol)

    # Get RMSE
    # if symmetric:
    #    print("wRMSE: ", (((test_output - mean[:, 0:int(N_tasks / 2)]) ** 2) / (test_noise)).mean().sqrt().item())
    # else:
    #    print("wRMSE: ", (((test_output - mean) ** 2) / (test_noise)).mean().sqrt().item())

    # Write output to file
    pd.DataFrame(mean.numpy()).to_csv(os.path.join("data", "output.csv"))

    # Done!
    print("Script is done!")




