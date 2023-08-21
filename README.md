[![DOI](https://zenodo.org/badge/465453049.svg)](https://zenodo.org/badge/latestdoi/465453049)

# PIICM
Permutation Invariant multi-output Gaussian Process regression for dose-response prediction


## Installation
1. Set up a conda environment and install [PyTorch](https://pytorch.org/)
2. Install the modified version of [GPyTorch](https://gpytorch.ai/) by running 
```{python}
conda create --name piicm python=3.9
conda activate piicm
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
git clone https://github.com/ltronneb/PIICM.git
cd PIICM/gpytorch
pip install .
```
3. Scripts for pre-processing the dataset are found in preprocessing/
4. Script for formatting the data and running models are in model/
